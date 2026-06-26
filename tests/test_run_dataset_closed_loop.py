import json
from pathlib import Path

from scripts import run_dataset_closed_loop as closed_loop


def _make_dataset_project(root: Path) -> None:
    scripts = root / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "build_dataset.py").write_text("print('build')\n", encoding="utf-8")


def test_build_plan_wires_closed_loop_steps_with_vlm_start(tmp_path):
    dataset_root = tmp_path / "dataset"
    _make_dataset_project(dataset_root)
    source_pdf = dataset_root / "inputs" / "pdfs" / "600004075219.pdf"
    source_pdf.parent.mkdir(parents=True, exist_ok=True)
    source_pdf.write_bytes(b"%PDF-1.4")

    args = closed_loop.build_arg_parser().parse_args(
        [
            "--source-pdf",
            str(source_pdf),
            "--dataset-project-root",
            str(dataset_root),
            "--output-root",
            str(tmp_path / "closed-loop"),
            "--text-only-count",
            "2",
            "--seed",
            "123",
            "--target-dpi",
            "144",
            "--limit",
            "1",
        ]
    )

    plan = closed_loop.build_plan(args)

    assert [step.name for step in plan.steps] == [
        "generate_dataset",
        "convert_dataset",
        "ensure_vlm",
        "evaluate_dataset",
    ]
    assert plan.generated_dataset_dir == tmp_path / "closed-loop" / "dataset_project_output"
    assert plan.evaluation_dataset_dir == tmp_path / "closed-loop" / "evaluation_dataset"
    assert plan.evaluation_output_dir == tmp_path / "closed-loop" / "evaluation"
    assert "--text-only-count" in plan.steps[0].command
    assert "2" in plan.steps[0].command
    assert "--target-dpi" in plan.steps[1].command
    assert "144" in plan.steps[1].command
    assert plan.steps[2].command == [str(closed_loop.PROJECT_ROOT / "scripts" / "start_vllm.sh")]
    assert "--limit" in plan.steps[3].command
    assert "1" in plan.steps[3].command
    assert plan.steps[3].env
    assert plan.steps[3].env["LLM_PROVIDER"] == "vllm"


def test_build_plan_can_skip_vlm_start(tmp_path):
    dataset_root = tmp_path / "dataset"
    _make_dataset_project(dataset_root)
    source_pdf = dataset_root / "inputs" / "pdfs" / "600004075219.pdf"
    source_pdf.parent.mkdir(parents=True, exist_ok=True)
    source_pdf.write_bytes(b"%PDF-1.4")

    args = closed_loop.build_arg_parser().parse_args(
        [
            "--source-pdf",
            str(source_pdf),
            "--dataset-project-root",
            str(dataset_root),
            "--output-root",
            str(tmp_path / "closed-loop"),
            "--skip-vlm-start",
        ]
    )

    plan = closed_loop.build_plan(args)

    assert [step.name for step in plan.steps] == [
        "generate_dataset",
        "convert_dataset",
        "evaluate_dataset",
    ]
    assert plan.steps[2].env is None


def test_dry_run_writes_closed_loop_summary(tmp_path):
    dataset_root = tmp_path / "dataset"
    _make_dataset_project(dataset_root)
    source_dir = dataset_root / "inputs" / "pdfs"
    source_dir.mkdir(parents=True, exist_ok=True)

    output_root = tmp_path / "closed-loop"
    args = closed_loop.build_arg_parser().parse_args(
        [
            "--source-dir",
            str(source_dir),
            "--dataset-project-root",
            str(dataset_root),
            "--output-root",
            str(output_root),
            "--dry-run",
        ]
    )

    summary = closed_loop.run_closed_loop(args)
    saved = json.loads((output_root / "closed_loop_summary.json").read_text(encoding="utf-8"))

    assert summary["success"] is True
    assert saved["dry_run"] is True
    assert saved["parameters"]["ensure_vlm"] is True
    assert saved["paths"]["generated_dataset"].endswith("dataset_project_output")
    assert saved["paths"]["evaluation_dataset"].endswith("evaluation_dataset")
    assert saved["paths"]["evaluation_output"].endswith("evaluation")


def test_generate_step_retries_with_incremented_seed(tmp_path, monkeypatch):
    generated_dir = tmp_path / "generated"
    seen_seeds: list[int] = []

    step = closed_loop.CommandStep(
        name="generate_dataset",
        command=["python", "build_dataset.py", "--seed", "10"],
        cwd=str(tmp_path),
        retryable_seed=True,
    )

    def fake_run_step(step: closed_loop.CommandStep):
        seed = int(step.command[step.command.index("--seed") + 1])
        seen_seeds.append(seed)
        if len(seen_seeds) < 3:
            generated_dir.mkdir(parents=True, exist_ok=True)
            raise closed_loop.StepFailedError(step=step, returncode=1)
        return {
            "name": step.name,
            "command": step.command,
            "cwd": step.cwd,
            "returncode": 0,
            "duration_seconds": 0.1,
        }

    monkeypatch.setattr(closed_loop, "run_step", fake_run_step)

    result = closed_loop.run_generate_step_with_retries(
        step,
        generated_dataset_dir=generated_dir,
        base_seed=10,
        attempts=4,
        seed_step=5,
    )

    assert seen_seeds == [10, 15, 20]
    assert result["seed"] == 20
    assert result["attempt_index"] == 3
    assert result["attempt_count"] == 4
    assert [item["seed"] for item in result["attempts"]] == [10, 15, 20]
