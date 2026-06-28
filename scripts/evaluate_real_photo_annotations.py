"""Evaluate manually reviewed real-photo desktop annotations."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize a real-photo dataset manifest exported from desktop annotations."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to real-photo dataset manifest.json.",
    )
    parser.add_argument(
        "--output",
        help="Optional output summary JSON path.",
    )
    return parser


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _count_box_decisions(sample: Mapping[str, Any]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for item in sample.get("predicted_boxes") or []:
        if not isinstance(item, Mapping):
            continue
        counter[str(item.get("decision") or "unreviewed")] += 1
    return counter


def _load_detection_duration(sample: Mapping[str, Any]) -> float | None:
    direct_value = sample.get("detection_duration_seconds")
    if direct_value is not None:
        try:
            return float(direct_value)
        except (TypeError, ValueError):
            pass

    run_dir = sample.get("run_dir")
    if not run_dir:
        return None
    for name in ("final_result.json", "result.json"):
        path = Path(str(run_dir)) / name
        if not path.exists():
            continue
        try:
            payload = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        value = payload.get("duration_seconds") or payload.get("detection_duration_seconds")
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None


def evaluate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    samples = [
        item
        for item in manifest.get("samples") or []
        if isinstance(item, Mapping)
    ]
    decision_counts: Counter[str] = Counter()
    by_code: dict[str, Counter[str]] = defaultdict(Counter)
    durations: list[float] = []
    manual_missing_count = 0
    reviewed_count = 0

    for sample in samples:
        code = str(sample.get("code") or "unknown")
        sample_decisions = _count_box_decisions(sample)
        decision_counts.update(sample_decisions)
        sample_missing_count = len(sample.get("manual_gt_boxes") or [])
        manual_missing_count += sample_missing_count
        sample_reviewed = (
            sample_decisions.get("true_positive", 0)
            + sample_decisions.get("false_positive", 0)
            + sample_missing_count
            > 0
        )
        reviewed_count += 1 if sample_reviewed else 0
        by_code[code]["reviewed" if sample_reviewed else "unreviewed"] += 1
        duration = _load_detection_duration(sample)
        if duration is not None:
            durations.append(duration)

    true_positive = int(decision_counts.get("true_positive", 0))
    false_positive = int(decision_counts.get("false_positive", 0))
    reviewed_predicted = true_positive + false_positive
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else None
    )
    recall_proxy = (
        true_positive / (true_positive + manual_missing_count)
        if true_positive + manual_missing_count
        else None
    )

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_name": manifest.get("dataset_name"),
        "sample_count": len(samples),
        "reviewed_count": reviewed_count,
        "box_decision_counts": dict(sorted(decision_counts.items())),
        "manual_missing_box_count": manual_missing_count,
        "reviewed_predicted_box_count": reviewed_predicted,
        "precision": round(precision, 4) if precision is not None else None,
        "recall_proxy": round(recall_proxy, 4) if recall_proxy is not None else None,
        "detection_duration_seconds": {
            "average": round(mean(durations), 3) if durations else None,
            "median": round(median(durations), 3) if durations else None,
        },
        "by_code": {
            code: dict(sorted(counter.items()))
            for code, counter in sorted(by_code.items())
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest_path = Path(args.manifest).resolve()
    summary = evaluate_manifest(read_json(manifest_path))
    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"summary={output_path}")
    print(
        "Result: "
        f"samples={summary['sample_count']} "
        f"reviewed={summary['reviewed_count']} "
        f"precision={summary['precision']} "
        f"recall_proxy={summary['recall_proxy']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
