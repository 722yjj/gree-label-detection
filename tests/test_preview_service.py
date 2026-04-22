import os
import time
from pathlib import Path

from desktop_app.models import TemplateRecord
from desktop_app.services import preview_service as preview_service_module
from desktop_app.services.preview_service import PreviewService


def make_pdf_template(tmp_path: Path, code: str = "600004075219") -> TemplateRecord:
    source_path = tmp_path / f"{code}.pdf"
    source_path.write_bytes(b"%PDF-1.4")
    return TemplateRecord(
        code=code,
        variant=None,
        display_name=source_path.stem,
        source_type="pdf",
        source_path=source_path,
    )


def make_preview(path: Path, *, mtime_ns: int | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"preview")
    if mtime_ns is not None:
        os.utime(path, ns=(mtime_ns, mtime_ns))
    return path


def test_resolve_preview_path_reuses_fresh_cached_pdf_preview(tmp_path, monkeypatch):
    template = make_pdf_template(tmp_path)
    source_mtime_ns = time.time_ns() - 5_000_000_000
    os.utime(template.source_path, ns=(source_mtime_ns, source_mtime_ns))

    cache_root = tmp_path / "cache"
    cached_preview = make_preview(
        cache_root / template.source_path.stem / f"{template.source_path.stem}_page1.png",
        mtime_ns=source_mtime_ns + 1_000_000,
    )

    service = PreviewService(cache_root=cache_root)

    def fail_resolve(*_args, **_kwargs):
        raise AssertionError("fresh cached preview should be reused")

    monkeypatch.setattr(preview_service_module, "resolve_template_input", fail_resolve)

    assert service.resolve_preview_path(template) == cached_preview.resolve()


def test_resolve_preview_path_regenerates_stale_preview_when_pdf_mtime_changes(
    tmp_path,
    monkeypatch,
):
    template = make_pdf_template(tmp_path)
    source_mtime_ns = time.time_ns()
    os.utime(template.source_path, ns=(source_mtime_ns, source_mtime_ns))

    cache_root = tmp_path / "cache"
    cache_dir = cache_root / template.source_path.stem
    stale_preview = make_preview(
        cache_dir / f"{template.source_path.stem}_page2.png",
        mtime_ns=source_mtime_ns - 5_000_000_000,
    )
    stale_text = cache_dir / f"{template.source_path.stem}_page2.txt"
    stale_text.write_text("stale", encoding="utf-8")
    os.utime(stale_text, ns=(source_mtime_ns - 5_000_000_000, source_mtime_ns - 5_000_000_000))

    calls: list[Path] = []

    def fake_resolve(template_path: str, output_dir: str | None = None, target_dpi: int = 300):
        assert target_dpi == 180
        calls.append(Path(template_path))
        output_path = Path(output_dir) / f"{Path(template_path).stem}_page1.png"
        output_path.write_bytes(b"fresh-preview")
        preview_mtime_ns = Path(template_path).stat().st_mtime_ns + 1_000_000
        os.utime(output_path, ns=(preview_mtime_ns, preview_mtime_ns))
        return str(output_path), "pdf", None

    monkeypatch.setattr(preview_service_module, "resolve_template_input", fake_resolve)

    service = PreviewService(cache_root=cache_root)
    resolved = service.resolve_preview_path(template)

    assert calls == [template.source_path.resolve()]
    assert resolved == (cache_dir / f"{template.source_path.stem}_page1.png").resolve()
    assert stale_preview.exists() is False
    assert stale_text.exists() is False


def test_resolve_preview_path_invalidates_memory_cache_on_pdf_mtime_change(
    tmp_path,
    monkeypatch,
):
    template = make_pdf_template(tmp_path)
    first_source_mtime_ns = time.time_ns() - 5_000_000_000
    os.utime(template.source_path, ns=(first_source_mtime_ns, first_source_mtime_ns))

    cache_root = tmp_path / "cache"
    call_count = 0

    def fake_resolve(template_path: str, output_dir: str | None = None, target_dpi: int = 300):
        nonlocal call_count
        call_count += 1
        output_path = Path(output_dir) / f"{Path(template_path).stem}_page{call_count}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(f"preview-{call_count}".encode("utf-8"))
        preview_mtime_ns = Path(template_path).stat().st_mtime_ns + 1_000_000
        os.utime(output_path, ns=(preview_mtime_ns, preview_mtime_ns))
        return str(output_path), "pdf", None

    monkeypatch.setattr(preview_service_module, "resolve_template_input", fake_resolve)

    service = PreviewService(cache_root=cache_root)
    first_resolved = service.resolve_preview_path(template)
    second_resolved = service.resolve_preview_path(template)

    updated_source_mtime_ns = time.time_ns()
    os.utime(template.source_path, ns=(updated_source_mtime_ns, updated_source_mtime_ns))
    third_resolved = service.resolve_preview_path(template)

    assert call_count == 2
    assert second_resolved == first_resolved
    assert third_resolved != first_resolved


def test_resolve_preview_path_returns_none_when_template_file_is_missing(tmp_path):
    missing_path = tmp_path / "missing.pdf"
    template = TemplateRecord(
        code="600004075219",
        variant=None,
        display_name=missing_path.stem,
        source_type="pdf",
        source_path=missing_path,
    )

    service = PreviewService(cache_root=tmp_path / "cache")

    assert service.resolve_preview_path(template) is None
