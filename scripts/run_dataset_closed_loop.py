"""Run the dataset-generation -> conversion -> evaluation closed loop."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PROJECT_ROOT = Path("/home/jnu/projects/dataset")
DEFAULT_OUTPUT_PARENT = PROJECT_ROOT / "results" / "dataset_closed_loop"


@dataclass(frozen=True)
class CommandStep:
    name: str
    command: list[str]
    cwd: str
    retryable_seed: bool = False
    env: dict[str, str] | None = None


@dataclass(frozen=True)
class ClosedLoopPlan:
    output_root: Path
    dataset_project_root: Path
    generated_dataset_dir: Path
    evaluation_dataset_dir: Path
    evaluation_output_dir: Path
    summary_path: Path
    steps: list[CommandStep]


class StepFailedError(RuntimeError):
    def __init__(
        self,
        *,
        step: CommandStep,
        returncode: int,
        attempts: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(f"step failed: {command_text(step.command)} (exit {returncode})")
        self.step = step
        self.returncode = int(returncode)
        self.attempts = list(attempts or [])


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full synthetic closed loop: dataset project generation, "
            "evaluation-dataset conversion, and detection evaluation."
        )
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--source-pdf",
        help="Source PDF passed to dataset/scripts/build_dataset.py.",
    )
    source_group.add_argument(
        "--source-dir",
        help="Directory of source PDFs passed to dataset/scripts/build_dataset.py.",
    )
    parser.add_argument(
        "--dataset-project-root",
        default=str(DEFAULT_DATASET_PROJECT_ROOT),
        help="Path to the dataset project checkout.",
    )
    parser.add_argument(
        "--output-root",
        help=(
            "Closed-loop output root. Defaults to "
            "results/dataset_closed_loop/<timestamp>."
        ),
    )
    parser.add_argument(
        "--text-only-count",
        type=int,
        default=3,
        help="Number of text-only samples generated per source PDF.",
    )
    parser.add_argument("--seed", type=int, default=20260625, help="Dataset build seed.")
    parser.add_argument(
        "--generate-retries",
        type=int,
        default=5,
        help=(
            "How many seed attempts to use for the dataset generation step. "
            "The dataset project can reject samples when mutations fail constraints."
        ),
    )
    parser.add_argument(
        "--seed-step",
        type=int,
        default=10007,
        help="Seed increment used between dataset generation retries.",
    )
    parser.add_argument(
        "--target-dpi",
        type=int,
        default=300,
        help="DPI used when converting generated PDFs to target PNGs.",
    )
    parser.add_argument(
        "--output-mode",
        choices=("final", "debug"),
        default="final",
        help="Detection workflow output mode.",
    )
    parser.add_argument(
        "--desktop-pipeline",
        choices=("traditional_full_image_diff", "unified"),
        default="traditional_full_image_diff",
        help="DetectionService pipeline used by the evaluation step.",
    )
    parser.add_argument(
        "--skip-vlm-start",
        action="store_true",
        help=(
            "Skip the desktop-launcher-style vLLM startup step before evaluation. "
            "By default the traditional desktop pipeline ensures vLLM is ready."
        ),
    )
    parser.add_argument(
        "--vllm-start-script",
        default=str(PROJECT_ROOT / "scripts" / "start_vllm.sh"),
        help="Path to the project vLLM startup script.",
    )
    parser.add_argument("--limit", type=int, help="Optional evaluation sample limit.")
    parser.add_argument(
        "--include-unstable-gt",
        action="store_true",
        help="Forward --include-unstable-gt to the evaluation step.",
    )
    parser.add_argument(
        "--dataset-python",
        help="Python executable for the dataset project. Defaults to dataset/.venv/bin/python.",
    )
    parser.add_argument(
        "--detector-python",
        help="Python executable for this project. Defaults to the current interpreter.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing closed-loop output root before running.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print and write the plan without running commands.",
    )
    return parser


def resolve_path(value: str, base: Path) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path.resolve()
    return (base / path).resolve()


def default_dataset_python(dataset_project_root: Path) -> Path:
    candidate = dataset_project_root / ".venv" / "bin" / "python"
    return candidate if candidate.exists() else Path(sys.executable)


def parse_export_lines(text: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("export ") or "=" not in stripped:
            continue
        assignment = stripped.removeprefix("export ").strip()
        key, value = assignment.split("=", 1)
        key = key.strip()
        if not key:
            continue
        env[key] = value.strip().strip("'\"")
    return env


def load_vllm_app_env(vllm_start_script: Path) -> dict[str, str]:
    if not vllm_start_script.exists():
        raise FileNotFoundError(f"vLLM startup script not found: {vllm_start_script}")
    completed = subprocess.run(
        [str(vllm_start_script), "--print-env"],
        cwd=str(PROJECT_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(
            f"Failed to read vLLM environment from {vllm_start_script}: {detail}"
        )
    env = parse_export_lines(completed.stdout)
    if "OPENAI_COMPATIBLE_API_BASE" not in env:
        raise RuntimeError(
            f"vLLM environment from {vllm_start_script} is missing OPENAI_COMPATIBLE_API_BASE"
        )
    return env


def should_ensure_vlm(args: argparse.Namespace) -> bool:
    if args.skip_vlm_start:
        return False
    return args.desktop_pipeline == "traditional_full_image_diff"


def prepare_output_root(path: Path, *, overwrite: bool, dry_run: bool) -> None:
    if dry_run:
        path.mkdir(parents=True, exist_ok=True)
        return
    if path.exists() and overwrite:
        shutil.rmtree(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(
            f"Closed-loop output root is not empty: {path}. "
            "Choose a new --output-root or pass --overwrite."
        )
    path.mkdir(parents=True, exist_ok=True)


def build_plan(args: argparse.Namespace) -> ClosedLoopPlan:
    dataset_project_root = Path(args.dataset_project_root).resolve()
    if not dataset_project_root.exists():
        raise FileNotFoundError(f"Dataset project root not found: {dataset_project_root}")

    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else (DEFAULT_OUTPUT_PARENT / timestamp_slug()).resolve()
    )
    generated_dataset_dir = output_root / "dataset_project_output"
    evaluation_dataset_dir = output_root / "evaluation_dataset"
    evaluation_output_dir = output_root / "evaluation"
    summary_path = output_root / "closed_loop_summary.json"

    dataset_python = (
        Path(args.dataset_python).resolve()
        if args.dataset_python
        else default_dataset_python(dataset_project_root)
    )
    detector_python = Path(args.detector_python).resolve() if args.detector_python else Path(sys.executable)

    build_dataset_script = dataset_project_root / "scripts" / "build_dataset.py"
    convert_script = PROJECT_ROOT / "scripts" / "prepare_dataset_project_eval_dataset.py"
    evaluate_script = PROJECT_ROOT / "scripts" / "evaluate_synthetic_dataset.py"
    vllm_start_script = Path(args.vllm_start_script).resolve()
    for script_path in (build_dataset_script, convert_script, evaluate_script):
        if not script_path.exists():
            raise FileNotFoundError(f"Required script not found: {script_path}")

    ensure_vlm = should_ensure_vlm(args)
    evaluation_env = load_vllm_app_env(vllm_start_script) if ensure_vlm else None

    generate_cmd = [
        str(dataset_python),
        str(build_dataset_script),
        "--output-dir",
        str(generated_dataset_dir),
        "--text-only-count",
        str(args.text_only_count),
        "--seed",
        str(args.seed),
    ]
    if args.source_pdf:
        generate_cmd.extend(
            ["--source-pdf", str(resolve_path(args.source_pdf, dataset_project_root))]
        )
    else:
        generate_cmd.extend(
            ["--source-dir", str(resolve_path(args.source_dir, dataset_project_root))]
        )

    convert_cmd = [
        str(detector_python),
        str(convert_script),
        "--source-root",
        str(generated_dataset_dir),
        "--output-dir",
        str(evaluation_dataset_dir),
        "--target-dpi",
        str(args.target_dpi),
        "--overwrite",
    ]

    evaluate_cmd = [
        str(detector_python),
        str(evaluate_script),
        "--dataset-root",
        str(evaluation_dataset_dir),
        "--output-dir",
        str(evaluation_output_dir),
        "--output-mode",
        args.output_mode,
        "--desktop-pipeline",
        args.desktop_pipeline,
    ]
    if args.limit is not None:
        evaluate_cmd.extend(["--limit", str(args.limit)])
    if args.include_unstable_gt:
        evaluate_cmd.append("--include-unstable-gt")

    steps = [
        CommandStep(
            "generate_dataset",
            generate_cmd,
            str(dataset_project_root),
            retryable_seed=True,
        ),
        CommandStep("convert_dataset", convert_cmd, str(PROJECT_ROOT)),
    ]
    if ensure_vlm:
        steps.append(
            CommandStep(
                "ensure_vlm",
                [str(vllm_start_script)],
                str(PROJECT_ROOT),
            )
        )
    steps.append(
        CommandStep(
            "evaluate_dataset",
            evaluate_cmd,
            str(PROJECT_ROOT),
            env=evaluation_env,
        )
    )

    return ClosedLoopPlan(
        output_root=output_root,
        dataset_project_root=dataset_project_root,
        generated_dataset_dir=generated_dataset_dir,
        evaluation_dataset_dir=evaluation_dataset_dir,
        evaluation_output_dir=evaluation_output_dir,
        summary_path=summary_path,
        steps=steps,
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def command_text(command: Sequence[str]) -> str:
    return " ".join(command)


def run_step(step: CommandStep) -> dict[str, Any]:
    print("=" * 72)
    print(f"[closed-loop] {step.name}")
    print(f"cwd: {step.cwd}")
    print(command_text(step.command))
    sys.stdout.flush()
    started = time.time()
    env = None
    if step.env:
        env = os.environ.copy()
        env.update(step.env)
    completed = subprocess.run(step.command, cwd=step.cwd, check=False, env=env)
    duration_seconds = round(time.time() - started, 3)
    record = {
        "name": step.name,
        "command": step.command,
        "cwd": step.cwd,
        "returncode": completed.returncode,
        "duration_seconds": duration_seconds,
    }
    if step.env:
        record["env"] = {
            key: value
            for key, value in step.env.items()
            if key
            in {
                "LLM_PROVIDER",
                "OPENAI_COMPATIBLE_API_BASE",
                "OPENAI_COMPATIBLE_MODEL",
                "VLLM_API_BASE",
                "VLLM_MODEL",
                "VLLM_SERVED_MODEL_NAME",
                "TEXT_LLM_MODEL",
                "GRAPHIC_VLM_MODEL",
            }
        }
    if completed.returncode != 0:
        raise StepFailedError(step=step, returncode=completed.returncode)
    return record


def command_with_seed(command: Sequence[str], seed: int) -> list[str]:
    updated = list(command)
    try:
        seed_index = updated.index("--seed")
    except ValueError:
        return updated
    if seed_index + 1 >= len(updated):
        return updated
    updated[seed_index + 1] = str(seed)
    return updated


def run_generate_step_with_retries(
    step: CommandStep,
    *,
    generated_dataset_dir: Path,
    base_seed: int,
    attempts: int,
    seed_step: int,
) -> dict[str, Any]:
    max_attempts = max(1, int(attempts))
    attempt_records: list[dict[str, Any]] = []
    last_error: StepFailedError | None = None

    for attempt_index in range(max_attempts):
        seed = int(base_seed) + attempt_index * int(seed_step)
        if generated_dataset_dir.exists():
            shutil.rmtree(generated_dataset_dir)
        attempt_step = CommandStep(
            name=step.name,
            command=command_with_seed(step.command, seed),
            cwd=step.cwd,
            retryable_seed=step.retryable_seed,
        )
        try:
            record = run_step(attempt_step)
            record["seed"] = seed
            record["attempt_index"] = attempt_index + 1
            record["attempt_count"] = max_attempts
            record["attempts"] = [
                *attempt_records,
                {
                    "name": attempt_step.name,
                    "command": attempt_step.command,
                    "cwd": attempt_step.cwd,
                    "returncode": record["returncode"],
                    "seed": seed,
                    "attempt_index": attempt_index + 1,
                    "attempt_count": max_attempts,
                    "duration_seconds": record["duration_seconds"],
                },
            ]
            return record
        except StepFailedError as exc:
            last_error = exc
            attempt_records.append(
                {
                    "name": step.name,
                    "command": attempt_step.command,
                    "cwd": attempt_step.cwd,
                    "returncode": exc.returncode,
                    "seed": seed,
                    "attempt_index": attempt_index + 1,
                    "attempt_count": max_attempts,
                }
            )
            print(
                f"[closed-loop] generation failed with seed={seed} "
                f"({attempt_index + 1}/{max_attempts})"
            )

    assert last_error is not None
    raise StepFailedError(
        step=last_error.step,
        returncode=last_error.returncode,
        attempts=attempt_records,
    )


def read_optional_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8-sig") as file:
        payload = json.load(file)
    return payload if isinstance(payload, dict) else None


def build_summary(
    plan: ClosedLoopPlan,
    *,
    args: argparse.Namespace,
    step_records: Sequence[dict[str, Any]],
    success: bool,
    error: Optional[str] = None,
) -> dict[str, Any]:
    evaluation_summary = read_optional_json(plan.evaluation_output_dir / "summary.json")
    return {
        "success": success,
        "error": error,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "parameters": {
            "source_pdf": args.source_pdf,
            "source_dir": args.source_dir,
            "text_only_count": args.text_only_count,
            "seed": args.seed,
            "generate_retries": args.generate_retries,
            "seed_step": args.seed_step,
            "target_dpi": args.target_dpi,
            "output_mode": args.output_mode,
            "desktop_pipeline": args.desktop_pipeline,
            "ensure_vlm": should_ensure_vlm(args),
            "vllm_start_script": args.vllm_start_script,
            "limit": args.limit,
            "include_unstable_gt": bool(args.include_unstable_gt),
        },
        "paths": {
            "output_root": str(plan.output_root),
            "generated_dataset": str(plan.generated_dataset_dir),
            "evaluation_dataset": str(plan.evaluation_dataset_dir),
            "evaluation_output": str(plan.evaluation_output_dir),
            "evaluation_summary": str(plan.evaluation_output_dir / "summary.json"),
        },
        "steps": list(step_records),
        "evaluation_stats": (
            evaluation_summary.get("stats") if evaluation_summary else None
        ),
    }


def print_plan(plan: ClosedLoopPlan) -> None:
    print("=" * 72)
    print("Dataset closed-loop plan")
    print("=" * 72)
    print(f"Output root: {plan.output_root}")
    for step in plan.steps:
        print(f"[{step.name}] cwd={step.cwd}")
        print(command_text(step.command))
    sys.stdout.flush()


def run_closed_loop(args: argparse.Namespace) -> dict[str, Any]:
    plan = build_plan(args)
    prepare_output_root(plan.output_root, overwrite=args.overwrite, dry_run=args.dry_run)
    print_plan(plan)

    if args.dry_run:
        summary = build_summary(plan, args=args, step_records=[], success=True)
        summary["dry_run"] = True
        write_json(plan.summary_path, summary)
        print(f"Dry-run summary: {plan.summary_path}")
        return summary

    step_records: list[dict[str, Any]] = []
    try:
        for step in plan.steps:
            if step.retryable_seed:
                step_records.append(
                    run_generate_step_with_retries(
                        step,
                        generated_dataset_dir=plan.generated_dataset_dir,
                        base_seed=args.seed,
                        attempts=args.generate_retries,
                        seed_step=args.seed_step,
                    )
                )
            else:
                step_records.append(run_step(step))
    except StepFailedError as exc:
        if exc.attempts:
            step_records.append(
                {
                    "name": exc.step.name,
                    "command": exc.step.command,
                    "cwd": exc.step.cwd,
                    "returncode": exc.returncode,
                    "attempts": exc.attempts,
                }
            )
        summary = build_summary(
            plan,
            args=args,
            step_records=step_records,
            success=False,
            error=str(exc),
        )
        write_json(plan.summary_path, summary)
        raise

    summary = build_summary(plan, args=args, step_records=step_records, success=True)
    write_json(plan.summary_path, summary)
    print("=" * 72)
    print(f"Closed-loop summary: {plan.summary_path}")
    if summary.get("evaluation_stats"):
        stats = summary["evaluation_stats"]
        print(
            "Evaluation: "
            f"passed={stats.get('passed')} "
            f"failed={stats.get('failed')} "
            f"total={stats.get('total')} "
            f"pass_rate={stats.get('pass_rate')}"
        )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        run_closed_loop(args)
    except StepFailedError as exc:
        return int(exc.returncode or 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
