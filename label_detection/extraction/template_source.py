"""Template input resolution helpers."""

from pathlib import Path
from typing import Optional, Tuple


SUPPORTED_TEMPLATE_IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def extract_red_box_info(*args, **kwargs):
    """Lazily import PDF extraction so image-only flows don't require fitz."""
    from label_detection.extraction.pdf import extract_red_box_info as _extract_red_box_info

    return _extract_red_box_info(*args, **kwargs)


def is_supported_template_image(path: str) -> bool:
    """Return True when the path looks like a supported template image."""
    return Path(path).suffix.lower() in SUPPORTED_TEMPLATE_IMAGE_SUFFIXES


def resolve_template_input(
    template_path: str,
    output_dir: Optional[str] = None,
    target_dpi: int = 300,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Resolve a template input into an image path consumable by the workflow.

    Returns:
        resolved_image_path: Path to the raw template image.
        source_type: "pdf" or "image".
        error: Error message when resolution fails.
    """
    template_file = Path(template_path)

    if not template_file.exists():
        return None, None, f"找不到模板文件: {template_path}"

    suffix = template_file.suffix.lower()
    if suffix == ".pdf":
        try:
            extracted_path = extract_red_box_info(
                str(template_file),
                target_dpi=target_dpi,
                output_dir=output_dir,
            )
        except ModuleNotFoundError as exc:
            return None, "pdf", f"PDF 模板解析依赖缺失: {exc.name}"
        if not extracted_path:
            return None, "pdf", "无法从 PDF 提取模板图片"
        return extracted_path, "pdf", None

    if suffix in SUPPORTED_TEMPLATE_IMAGE_SUFFIXES:
        return str(template_file), "image", None

    supported_types = ", ".join(
        [".pdf"] + sorted(SUPPORTED_TEMPLATE_IMAGE_SUFFIXES)
    )
    return (
        None,
        None,
        f"不支持的模板文件类型: {template_file.suffix or '(无扩展名)'}。"
        f" 当前支持: {supported_types}",
    )
