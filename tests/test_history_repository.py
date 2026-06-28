from pathlib import Path
import json

from desktop_app.models import HistoryRecord
from desktop_app.repositories.history_repository import HistoryRepository
from desktop_app.repositories.result_repository import DetectionResultRepository


def make_record(tmp_path: Path, *, suffix: str) -> HistoryRecord:
    output_dir = tmp_path / f"output-{suffix}"
    target_path = tmp_path / f"target-{suffix}.jpg"
    return HistoryRecord(
        created_at=f"2026-04-22T10:00:0{suffix}",
        code=f"60000407521{suffix}",
        template_name=f"template-{suffix}.pdf",
        verdict="需复核",
        output_dir=output_dir,
        target_image_path=target_path,
        visualization_path=output_dir / "visualization_diff.jpg",
        summary_text=f"summary-{suffix}",
    )


def test_history_repository_append_and_list_recent(tmp_path):
    repository = HistoryRepository(tmp_path / "history.json")
    record_a = make_record(tmp_path, suffix="1")
    record_b = make_record(tmp_path, suffix="2")

    repository.append(record_a)
    repository.append(record_b)

    records = repository.list_recent()

    assert [record.code for record in records] == [record_b.code, record_a.code]
    assert records[0].output_dir == record_b.output_dir
    assert records[0].visualization_path == record_b.visualization_path


def test_history_repository_returns_empty_list_for_invalid_json(tmp_path):
    history_file = tmp_path / "history.json"
    history_file.write_text("{not-json", encoding="utf-8")
    repository = HistoryRepository(history_file)

    assert repository.list_recent() == []


def test_detection_result_repository_rebuilds_result_from_history_run(tmp_path):
    record = make_record(tmp_path, suffix="1")
    record.output_dir.mkdir(parents=True)
    record.target_image_path.write_bytes(b"target")
    visualization = record.output_dir / "visualization_diff.jpg"
    visualization.write_bytes(b"vis")
    annotation = {
        "detection_duration_seconds": 4.2,
    }
    (record.output_dir / "manual_annotation.json").write_text(
        json.dumps(annotation, ensure_ascii=False),
        encoding="utf-8",
    )
    raw_result = {
        "success": True,
        "pipeline": "traditional_full_image_diff",
        "verdict": "不一致",
        "final_box_count": 2,
        "raw_candidate_box_count": 3,
        "merged_candidate_box_count": 2,
        "vlm_filter_results": [
            {"vlm_decision": "keep"},
            {"vlm_decision": "discard"},
        ],
        "final_boxes": [
            {"box": [1, 2, 3, 4]},
            {"box": [5, 6, 7, 8]},
        ],
        "visualization_diff": str(visualization),
    }
    (record.output_dir / "final_result.json").write_text(
        json.dumps(raw_result, ensure_ascii=False),
        encoding="utf-8",
    )

    restored = DetectionResultRepository().build_from_history_record(record)

    assert restored is not None
    result, duration = restored
    assert result.output_dir == record.output_dir
    assert result.verdict == "不一致"
    assert result.visualization_path == visualization
    assert result.graphic_mismatch_count == 2
    assert result.summary_text == "summary-1"
    assert duration == 4.2

    record_without_summary = HistoryRecord(
        created_at=record.created_at,
        code=record.code,
        template_name=record.template_name,
        verdict=record.verdict,
        output_dir=record.output_dir,
        target_image_path=record.target_image_path,
        visualization_path=record.visualization_path,
        summary_text="",
    )
    rebuilt, _duration = DetectionResultRepository().build_from_history_record(
        record_without_summary
    )
    assert rebuilt is not None
    assert "候选框: 3 -> 2" in rebuilt.summary_text
