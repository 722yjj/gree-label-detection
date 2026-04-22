"""JSON-backed history storage for desktop detection runs."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from desktop_app.models import HistoryRecord


class HistoryRepository:
    """Persist and load recent detection history for the desktop UI."""

    def __init__(self, history_file: Path) -> None:
        self.history_file = Path(history_file)

    def list_recent(self, limit: int = 20) -> list[HistoryRecord]:
        records = self._load_records()
        return records[:limit]

    def append(self, record: HistoryRecord) -> None:
        records = self._load_records()
        records.insert(0, record)
        self._write_records(records)

    def _load_records(self) -> list[HistoryRecord]:
        if not self.history_file.exists():
            return []

        try:
            raw = json.loads(self.history_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[HistoryRepository] failed to read {self.history_file}: {exc}")
            return []

        if not isinstance(raw, list):
            return []

        records: list[HistoryRecord] = []
        for item in raw:
            try:
                records.append(self._record_from_json(item))
            except (KeyError, TypeError, ValueError) as exc:
                print(f"[HistoryRepository] skipped invalid history entry: {exc}")
        return records

    def _write_records(self, records: list[HistoryRecord]) -> None:
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        payload = [self._record_to_json(record) for record in records]
        self.history_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _record_to_json(record: HistoryRecord) -> dict[str, object]:
        payload = asdict(record)
        payload["output_dir"] = str(record.output_dir)
        payload["target_image_path"] = str(record.target_image_path)
        payload["visualization_path"] = (
            str(record.visualization_path) if record.visualization_path else None
        )
        return payload

    @staticmethod
    def _record_from_json(payload: dict[str, object]) -> HistoryRecord:
        return HistoryRecord(
            created_at=str(payload["created_at"]),
            code=str(payload["code"]),
            template_name=str(payload["template_name"]),
            verdict=str(payload["verdict"]),
            output_dir=Path(str(payload["output_dir"])),
            target_image_path=Path(str(payload["target_image_path"])),
            visualization_path=(
                Path(str(payload["visualization_path"]))
                if payload.get("visualization_path")
                else None
            ),
            summary_text=str(payload.get("summary_text") or ""),
        )
