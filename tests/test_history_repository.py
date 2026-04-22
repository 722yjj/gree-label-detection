from pathlib import Path

from desktop_app.models import HistoryRecord
from desktop_app.repositories.history_repository import HistoryRepository


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
