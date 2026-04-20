"""Batch runner for sample-based template/target comparisons."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence
from uuid import uuid4

from label_detection.core.config import PROJECT_ROOT, SAMPLES_DIR
from label_detection.extraction.template_source import SUPPORTED_TEMPLATE_IMAGE_SUFFIXES


DEFAULT_TEMPLATE_ROOTS = (SAMPLES_DIR / "pdfs",)
DEFAULT_TARGET_ROOTS = (SAMPLES_DIR / "images" / "produce",)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "batch_samples"
TARGET_IMAGE_SUFFIXES = tuple(sorted(SUPPORTED_TEMPLATE_IMAGE_SUFFIXES))
TEMPLATE_SUFFIXES = (".pdf",) + TARGET_IMAGE_SUFFIXES

RESULT_JSON_NAME = "result.json"
RESULT_VIS_NAME = "visualization_diff.jpg"
SUMMARY_JSON_NAME = "summary.json"

_RUN_UNIFIED_DETECTION = None


@dataclass(frozen=True)
class SampleAsset:
    path: Path
    code: str
    stem: str


@dataclass(frozen=True)
class BatchCase:
    code: str
    template: SampleAsset
    target: SampleAsset
    case_id: str


@dataclass
class CaseRunRecord:
    code: str
    case_id: str
    template_path: str
    target_path: str
    result_json: str
    visualization_diff: Optional[str]
    success: bool
    skipped: bool
    verdict: Optional[str]
    error: Optional[str]
    duration_seconds: float


@dataclass
class DiscoveryResult:
    templates_by_code: Dict[str, List[SampleAsset]]
    targets_by_code: Dict[str, List[SampleAsset]]
    ignored_templates: List[str]
    ignored_targets: List[str]


def get_detection_runner():
    """Import the heavy workflow lazily so discovery can be tested cheaply."""
    global _RUN_UNIFIED_DETECTION
    if _RUN_UNIFIED_DETECTION is None:
        from label_detection.workflows.unified import run_unified_detection

        _RUN_UNIFIED_DETECTION = run_unified_detection
    return _RUN_UNIFIED_DETECTION


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def normalize_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return normalized.strip("._-") or "unnamed"


def extract_label_code(name: str, min_code_length: int = 6) -> Optional[str]:
    match = re.search(rf"(\d{{{min_code_length},}})", name)
    if not match:
        return None
    return match.group(1)


def iter_candidate_files(
    roots: Sequence[Path],
    suffixes: Sequence[str],
) -> Iterable[Path]:
    allowed = {suffix.lower() for suffix in suffixes}
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in allowed:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield resolved


def group_assets_by_code(
    roots: Sequence[Path],
    suffixes: Sequence[str],
    min_code_length: int,
) -> tuple[Dict[str, List[SampleAsset]], List[str]]:
    grouped: Dict[str, List[SampleAsset]] = {}
    ignored: List[str] = []

    for path in iter_candidate_files(roots, suffixes):
        code = extract_label_code(path.stem, min_code_length=min_code_length)
        if not code:
            ignored.append(repo_relative(path))
            continue

        asset = SampleAsset(path=path, code=code, stem=path.stem)
        grouped.setdefault(code, []).append(asset)

    for assets in grouped.values():
        assets.sort(key=lambda item: (item.stem != item.code, item.stem, item.path.name))

    return grouped, ignored


def discover_samples(
    template_roots: Sequence[Path],
    target_roots: Sequence[Path],
    min_code_length: int = 6,
) -> DiscoveryResult:
    templates_by_code, ignored_templates = group_assets_by_code(
        template_roots,
        TEMPLATE_SUFFIXES,
        min_code_length,
    )
    targets_by_code, ignored_targets = group_assets_by_code(
        target_roots,
        TARGET_IMAGE_SUFFIXES,
        min_code_length,
    )
    return DiscoveryResult(
        templates_by_code=templates_by_code,
        targets_by_code=targets_by_code,
        ignored_templates=ignored_templates,
        ignored_targets=ignored_targets,
    )


def choose_preferred_template(code: str, templates: Sequence[SampleAsset]) -> SampleAsset:
    def sort_key(asset: SampleAsset) -> tuple[int, int, str]:
        if asset.stem == code:
            variant_rank = 0
        elif asset.stem.startswith(f"{code}-"):
            variant_rank = 1
        elif asset.stem.startswith(f"{code}_"):
            variant_rank = 2
        else:
            variant_rank = 3
        pdf_rank = 0 if asset.path.suffix.lower() == ".pdf" else 1
        return (variant_rank, pdf_rank, asset.stem)

    return min(templates, key=sort_key)


def build_case_id(template: SampleAsset, target: SampleAsset) -> str:
    return normalize_name(f"{template.stem}__vs__{target.stem}")


def build_cases(
    templates_by_code: Dict[str, List[SampleAsset]],
    targets_by_code: Dict[str, List[SampleAsset]],
    pair_mode: str = "all",
    only_codes: Optional[Sequence[str]] = None,
) -> List[BatchCase]:
    allowed_codes = set(only_codes or [])
    shared_codes = sorted(set(templates_by_code) & set(targets_by_code))
    if allowed_codes:
        shared_codes = [code for code in shared_codes if code in allowed_codes]

    cases: List[BatchCase] = []
    for code in shared_codes:
        templates = templates_by_code[code]
        targets = targets_by_code[code]
        if pair_mode == "best-template":
            templates = [choose_preferred_template(code, templates)]

        for template in templates:
            for target in targets:
                cases.append(
                    BatchCase(
                        code=code,
                        template=template,
                        target=target,
                        case_id=build_case_id(template, target),
                    )
                )

    return cases


def compute_text_summary(result: Dict[str, object]) -> Dict[str, object]:
    text_detection = dict(result.get("text_detection") or {})
    template_data = dict(text_detection.get("template_data") or {})
    target_data = dict(text_detection.get("target_data") or {})
    field_names = list(text_detection.get("fields") or template_data.keys() or target_data.keys())
    different_fields = [
        field_name
        for field_name in field_names
        if template_data.get(field_name) != target_data.get(field_name)
    ]

    summary = {
        "label_kind": text_detection.get("label_kind"),
        "match_count": len(field_names) - len(different_fields),
        "total_fields": len(field_names),
        "different_fields": different_fields,
        "template_data": template_data,
        "target_data": target_data,
    }
    elapsed = text_detection.get("time")
    if elapsed is not None:
        summary["time"] = elapsed
    return summary


def compute_graphic_summary(result: Dict[str, object]) -> Dict[str, object]:
    graphic = dict(result.get("graphic_comparison") or {})
    comparison_results = list(graphic.get("comparison_results") or [])
    mismatch_count = sum(
        1
        for item in comparison_results
        if item.get("decision") == "mismatch" and not item.get("needs_review", False)
    )
    review_count = sum(
        1
        for item in comparison_results
        if item.get("decision") == "unknown" or item.get("needs_review", False)
    )
    matched_count = int(graphic.get("matched_count") or 0)
    recovered_match_count = int(graphic.get("recovered_match_count") or 0)
    resolved_match_count = int(
        graphic.get("resolved_match_count") or (matched_count + recovered_match_count)
    )

    return {
        "template_regions_count": graphic.get("template_regions_count"),
        "target_regions_count": graphic.get("target_regions_count"),
        "matched_count": matched_count,
        "recovered_match_count": recovered_match_count,
        "resolved_match_count": resolved_match_count,
        "effective_matched_count": resolved_match_count,
        "remaining_unmatched_template": list(graphic.get("remaining_unmatched_template") or []),
        "remaining_unmatched_target": list(graphic.get("remaining_unmatched_target") or []),
        "mismatch_count": mismatch_count,
        "review_count": review_count,
        "comparison_results": comparison_results,
    }


def build_saved_payload(
    case: BatchCase,
    raw_result: Dict[str, object],
    result_json_path: Path,
    visualization_path: Optional[Path],
    duration_seconds: float,
) -> Dict[str, object]:
    payload = {
        "success": bool(raw_result.get("success", True)),
        "code": case.code,
        "case_id": case.case_id,
        "template": {
            "path": repo_relative(case.template.path),
            "stem": case.template.stem,
        },
        "target": {
            "path": repo_relative(case.target.path),
            "stem": case.target.stem,
        },
        "verdict": raw_result.get("verdict"),
        "duration_seconds": round(duration_seconds, 3),
        "artifacts": {
            "result_json": repo_relative(result_json_path),
            "visualization_diff": repo_relative(visualization_path) if visualization_path else None,
        },
        "text_detection": compute_text_summary(raw_result),
        "graphic_comparison": compute_graphic_summary(raw_result),
    }

    template_input = dict(raw_result.get("template_input") or {})
    if template_input:
        payload["template_input"] = {
            "source_path": template_input.get("source_path") or repo_relative(case.template.path),
            "source_type": template_input.get("source_type"),
        }

    if not payload["success"]:
        payload["error"] = raw_result.get("error")

    return payload


def load_json_if_exists(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def get_case_output_dir(output_root: Path, case: BatchCase) -> Path:
    return output_root / case.code / case.case_id


def load_existing_record(case: BatchCase, output_root: Path) -> Optional[CaseRunRecord]:
    case_dir = get_case_output_dir(output_root, case)
    result_json_path = case_dir / RESULT_JSON_NAME
    if not result_json_path.exists():
        return None

    try:
        payload = load_json_if_exists(result_json_path)
    except json.JSONDecodeError:
        return None

    visualization_path = case_dir / RESULT_VIS_NAME
    return CaseRunRecord(
        code=case.code,
        case_id=case.case_id,
        template_path=repo_relative(case.template.path),
        target_path=repo_relative(case.target.path),
        result_json=repo_relative(result_json_path),
        visualization_diff=repo_relative(visualization_path) if visualization_path.exists() else None,
        success=bool(payload.get("success")),
        skipped=True,
        verdict=payload.get("verdict"),
        error=payload.get("error"),
        duration_seconds=float(payload.get("duration_seconds") or 0.0),
    )


def run_case(case: BatchCase, output_root: Path) -> CaseRunRecord:
    case_dir = get_case_output_dir(output_root, case)
    case_dir.mkdir(parents=True, exist_ok=True)

    result_json_path = case_dir / RESULT_JSON_NAME
    visualization_path = case_dir / RESULT_VIS_NAME
    if result_json_path.exists():
        result_json_path.unlink()
    if visualization_path.exists():
        visualization_path.unlink()

    started_at = time.time()
    runner = get_detection_runner()
    temp_root = output_root / "_tmp"
    temp_root.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, object]
    final_visualization: Optional[Path] = None
    error_message: Optional[str] = None
    work_dir = temp_root / f"{case.case_id}_{uuid4().hex}"

    try:
        work_dir.mkdir(parents=True, exist_ok=False)
        raw_result = runner(
            str(case.template.path),
            str(case.target.path),
            output_dir=str(work_dir),
        )
        raw_result = load_json_if_exists(work_dir / "final_result.json") or dict(raw_result or {})

        temp_visualization = work_dir / "visualization_diff.jpg"
        if temp_visualization.exists():
            shutil.copy2(temp_visualization, visualization_path)
            final_visualization = visualization_path

        payload = build_saved_payload(
            case,
            raw_result=raw_result,
            result_json_path=result_json_path,
            visualization_path=final_visualization,
            duration_seconds=time.time() - started_at,
        )
    except Exception as exc:  # noqa: BLE001
        error_message = str(exc)
        payload = {
            "success": False,
            "code": case.code,
            "case_id": case.case_id,
            "template": {
                "path": repo_relative(case.template.path),
                "stem": case.template.stem,
            },
            "target": {
                "path": repo_relative(case.target.path),
                "stem": case.target.stem,
            },
            "verdict": None,
            "duration_seconds": round(time.time() - started_at, 3),
            "error": error_message,
            "artifacts": {
                "result_json": repo_relative(result_json_path),
                "visualization_diff": None,
            },
        }
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    write_json(result_json_path, payload)

    return CaseRunRecord(
        code=case.code,
        case_id=case.case_id,
        template_path=repo_relative(case.template.path),
        target_path=repo_relative(case.target.path),
        result_json=repo_relative(result_json_path),
        visualization_diff=repo_relative(final_visualization) if final_visualization else None,
        success=bool(payload.get("success")),
        skipped=False,
        verdict=payload.get("verdict"),
        error=payload.get("error") or error_message,
        duration_seconds=float(payload.get("duration_seconds") or 0.0),
    )


def build_summary_payload(
    discovery: DiscoveryResult,
    cases: Sequence[BatchCase],
    records: Sequence[CaseRunRecord],
    template_roots: Sequence[Path],
    target_roots: Sequence[Path],
    output_dir: Path,
    pair_mode: str,
    dry_run: bool,
    skip_existing: bool,
) -> Dict[str, object]:
    template_count = sum(len(items) for items in discovery.templates_by_code.values())
    target_count = sum(len(items) for items in discovery.targets_by_code.values())
    matched_codes = sorted(set(discovery.templates_by_code) & set(discovery.targets_by_code))
    success_count = sum(1 for record in records if record.success)
    failure_count = sum(1 for record in records if not record.success)
    skipped_count = sum(1 for record in records if record.skipped)

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": repo_relative(PROJECT_ROOT),
        "template_roots": [repo_relative(path) for path in template_roots],
        "target_roots": [repo_relative(path) for path in target_roots],
        "output_dir": repo_relative(output_dir),
        "pair_mode": pair_mode,
        "dry_run": dry_run,
        "skip_existing": skip_existing,
        "stats": {
            "template_file_count": template_count,
            "target_file_count": target_count,
            "template_code_count": len(discovery.templates_by_code),
            "target_code_count": len(discovery.targets_by_code),
            "matched_code_count": len(matched_codes),
            "case_count": len(cases),
            "success_count": success_count,
            "failure_count": failure_count,
            "skipped_count": skipped_count,
        },
        "ignored_templates": discovery.ignored_templates,
        "ignored_targets": discovery.ignored_targets,
        "template_only_codes": sorted(set(discovery.templates_by_code) - set(discovery.targets_by_code)),
        "target_only_codes": sorted(set(discovery.targets_by_code) - set(discovery.templates_by_code)),
        "cases": [asdict(record) for record in records],
    }


def print_discovery_summary(discovery: DiscoveryResult, cases: Sequence[BatchCase]) -> None:
    template_count = sum(len(items) for items in discovery.templates_by_code.values())
    target_count = sum(len(items) for items in discovery.targets_by_code.values())
    matched_codes = sorted(set(discovery.templates_by_code) & set(discovery.targets_by_code))

    print("=" * 70)
    print("Batch sample detection")
    print("=" * 70)
    print(f"Templates found: {template_count} files / {len(discovery.templates_by_code)} codes")
    print(f"Targets found:   {target_count} files / {len(discovery.targets_by_code)} codes")
    print(f"Matched codes:   {len(matched_codes)}")
    print(f"Planned cases:   {len(cases)}")
    if discovery.ignored_templates:
        print(f"Ignored templates without code: {len(discovery.ignored_templates)}")
    if discovery.ignored_targets:
        print(f"Ignored targets without code:   {len(discovery.ignored_targets)}")


def parse_roots(values: Optional[Sequence[str]], defaults: Sequence[Path]) -> List[Path]:
    if not values:
        return [path.resolve() for path in defaults]
    return [Path(value).resolve() for value in values]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch test samples by matching template and target codes.")
    parser.add_argument(
        "--template-root",
        action="append",
        help="Template root directory. Repeat to scan multiple roots.",
    )
    parser.add_argument(
        "--target-root",
        action="append",
        help="Target image root directory. Repeat to scan multiple roots.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory used to store batch results.",
    )
    parser.add_argument(
        "--pair-mode",
        choices=("all", "best-template"),
        default="all",
        help="all = every template variant x every target variant; best-template = best template variant x every target variant.",
    )
    parser.add_argument(
        "--code",
        action="append",
        help="Only run the specified label code. Repeat to include multiple codes.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Only run the first N planned cases after sorting.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a case when its result.json already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print discovered cases without running the workflow.",
    )
    parser.add_argument(
        "--min-code-length",
        type=int,
        default=6,
        help="Minimum digit length used to extract the label code from filenames.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    template_roots = parse_roots(args.template_root, DEFAULT_TEMPLATE_ROOTS)
    target_roots = parse_roots(args.target_root, DEFAULT_TARGET_ROOTS)
    output_dir = Path(args.output_dir).resolve()

    discovery = discover_samples(
        template_roots=template_roots,
        target_roots=target_roots,
        min_code_length=args.min_code_length,
    )
    cases = build_cases(
        discovery.templates_by_code,
        discovery.targets_by_code,
        pair_mode=args.pair_mode,
        only_codes=args.code,
    )

    if args.limit is not None:
        cases = cases[: max(args.limit, 0)]

    print_discovery_summary(discovery, cases)
    for index, case in enumerate(cases, start=1):
        print(
            f"[plan {index:03d}] code={case.code} "
            f"template={repo_relative(case.template.path)} "
            f"target={repo_relative(case.target.path)}"
        )

    if args.dry_run:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    records: List[CaseRunRecord] = []

    for index, case in enumerate(cases, start=1):
        if args.skip_existing:
            existing = load_existing_record(case, output_dir)
            if existing is not None:
                print(f"[skip {index:03d}/{len(cases):03d}] {case.case_id}")
                records.append(existing)
                continue

        print(f"[run  {index:03d}/{len(cases):03d}] {case.case_id}")
        record = run_case(case, output_dir)
        records.append(record)
        if record.success:
            print(
                f"         success verdict={record.verdict or 'n/a'} "
                f"json={record.result_json}"
            )
        else:
            print(f"         failed error={record.error or 'unknown'}")

    summary = build_summary_payload(
        discovery=discovery,
        cases=cases,
        records=records,
        template_roots=template_roots,
        target_roots=target_roots,
        output_dir=output_dir,
        pair_mode=args.pair_mode,
        dry_run=args.dry_run,
        skip_existing=args.skip_existing,
    )
    summary_path = output_dir / SUMMARY_JSON_NAME
    write_json(summary_path, summary)
    temp_root = output_dir / "_tmp"
    if temp_root.exists() and not any(temp_root.iterdir()):
        temp_root.rmdir()

    print("=" * 70)
    print(f"Summary saved: {repo_relative(summary_path)}")
    print(
        "Run finished: "
        f"success={summary['stats']['success_count']} "
        f"failure={summary['stats']['failure_count']} "
        f"skipped={summary['stats']['skipped_count']}"
    )
    return 0 if summary["stats"]["failure_count"] == 0 else 1
