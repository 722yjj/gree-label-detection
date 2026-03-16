from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


Rect = Tuple[float, float, float, float]

_CONTAINER_SCALE = 1.15
_CONTAINER_AREA_SCALE = 1.35
_MAX_CONTAINER_PAGE_RATIO = 0.75
_CONTAIN_TOLERANCE = 1.5
_MIN_TEXT_BLOCKS_IN_CONTAINER = 3


def _load_fitz():
    import fitz  # PyMuPDF

    return fitz


def _normalize_rect(rect_like) -> Optional[Rect]:
    if rect_like is None:
        return None

    if hasattr(rect_like, "x0") and hasattr(rect_like, "y0"):
        x0 = float(rect_like.x0)
        y0 = float(rect_like.y0)
        x1 = float(rect_like.x1)
        y1 = float(rect_like.y1)
    else:
        try:
            x0, y0, x1, y1 = rect_like
        except (TypeError, ValueError):
            return None
        x0 = float(x0)
        y0 = float(y0)
        x1 = float(x1)
        y1 = float(y1)

    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0

    if x1 <= x0 or y1 <= y0:
        return None

    return (x0, y0, x1, y1)


def _rect_width(rect: Rect) -> float:
    return rect[2] - rect[0]


def _rect_height(rect: Rect) -> float:
    return rect[3] - rect[1]


def _rect_area(rect: Rect) -> float:
    return _rect_width(rect) * _rect_height(rect)


def _rect_contains(outer: Rect, inner: Rect, tolerance: float = _CONTAIN_TOLERANCE) -> bool:
    return (
        outer[0] <= inner[0] + tolerance
        and outer[1] <= inner[1] + tolerance
        and outer[2] >= inner[2] - tolerance
        and outer[3] >= inner[3] - tolerance
    )


def _union_rects(rects: Iterable[Rect]) -> Optional[Rect]:
    rect_list = [rect for rect in rects if rect is not None]
    if not rect_list:
        return None

    x0 = min(rect[0] for rect in rect_list)
    y0 = min(rect[1] for rect in rect_list)
    x1 = max(rect[2] for rect in rect_list)
    y1 = max(rect[3] for rect in rect_list)
    return (x0, y0, x1, y1)


def _is_red(color: Optional[Sequence[float]]) -> bool:
    return bool(color) and len(color) >= 3 and color[0] > 0.8 and color[1] < 0.2 and color[2] < 0.2


def _is_dashed(dashes) -> bool:
    if dashes is None:
        return False
    if isinstance(dashes, str):
        return dashes.strip() not in {"", "[]", "[] 0", "[]0", "[] 0 d"}
    return bool(dashes)


def _extract_image_block_rects(page) -> List[Rect]:
    blocks = page.get_text("dict").get("blocks", [])
    image_rects: List[Rect] = []

    for block in blocks:
        if block.get("type") != 1:
            continue
        rect = _normalize_rect(block.get("bbox"))
        if rect is not None:
            image_rects.append(rect)

    return image_rects


def _extract_text_block_rects(page) -> List[Rect]:
    blocks = page.get_text("dict").get("blocks", [])
    text_rects: List[Rect] = []

    for block in blocks:
        if block.get("type") != 0:
            continue
        rect = _normalize_rect(block.get("bbox"))
        if rect is None:
            continue
        text = "".join(
            span.get("text", "")
            for line in block.get("lines", [])
            for span in line.get("spans", [])
        ).strip()
        if text:
            text_rects.append(rect)

    return text_rects


def _extract_drawing_rects(page) -> List[Rect]:
    deduped: Dict[Tuple[float, float, float, float], Rect] = {}

    for path in page.get_drawings():
        rect = _normalize_rect(path.get("rect"))
        if rect is None:
            continue
        key = tuple(round(value, 2) for value in rect)
        deduped.setdefault(key, rect)

    return list(deduped.values())


def _find_smallest_container(
    image_rect: Rect,
    candidate_rects: Sequence[Rect],
    page_rect: Optional[Rect],
    text_rects: Sequence[Rect],
) -> Optional[Rect]:
    image_w = _rect_width(image_rect)
    image_h = _rect_height(image_rect)
    image_area = _rect_area(image_rect)
    page_area = _rect_area(page_rect) if page_rect else None

    valid_candidates: List[Rect] = []
    for candidate in candidate_rects:
        if not _rect_contains(candidate, image_rect):
            continue
        if _rect_width(candidate) < image_w * _CONTAINER_SCALE:
            continue
        if _rect_height(candidate) < image_h * _CONTAINER_SCALE:
            continue
        if _rect_area(candidate) <= image_area * _CONTAINER_AREA_SCALE:
            continue
        if page_area and (_rect_area(candidate) / page_area) > _MAX_CONTAINER_PAGE_RATIO:
            continue
        text_block_count = sum(
            1 for text_rect in text_rects if _rect_contains(candidate, text_rect)
        )
        if text_rects and text_block_count < _MIN_TEXT_BLOCKS_IN_CONTAINER:
            continue
        valid_candidates.append(candidate)

    if not valid_candidates:
        return None

    return min(valid_candidates, key=_rect_area)


def _select_image_container_rect(page) -> Optional[Rect]:
    image_rects = _extract_image_block_rects(page)
    if not image_rects:
        return None

    candidate_rects = _extract_drawing_rects(page)
    if not candidate_rects:
        return None

    page_rect = _normalize_rect(getattr(page, "rect", None))
    text_rects = _extract_text_block_rects(page)
    best_rect = None
    best_area = None

    for image_rect in image_rects:
        container = _find_smallest_container(
            image_rect,
            candidate_rects,
            page_rect,
            text_rects,
        )
        if container is None:
            continue
        area = _rect_area(container)
        if best_rect is None or area < best_area:
            best_rect = container
            best_area = area

    return best_rect


def _select_red_dashed_union_rect(page) -> Optional[Rect]:
    target_rects: List[Rect] = []

    for path in page.get_drawings():
        if not _is_red(path.get("color")) or not _is_dashed(path.get("dashes")):
            continue
        rect = _normalize_rect(path.get("rect"))
        if rect is not None:
            target_rects.append(rect)

    return _union_rects(target_rects)


def _select_page_crop_rect(page) -> Tuple[Optional[Rect], Optional[str]]:
    image_container = _select_image_container_rect(page)
    if image_container is not None:
        return image_container, "image_container"

    red_dashed_union = _select_red_dashed_union_rect(page)
    if red_dashed_union is not None:
        return red_dashed_union, "red_dashed_union"

    return None, None


def extract_red_box_info(pdf_path, target_dpi=300, output_dir=None):
    """
    Extract the template region from a PDF page as an image.

    Priority:
    1. The smallest drawing box that fully contains a PDF image block.
    2. Fallback to the union of dashed red drawing boxes.
    """
    fitz = _load_fitz()
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        print(f"错误：找不到文件 {pdf_path}")
        return None

    output_root = Path(output_dir) if output_dir else pdf_path.parent
    output_root.mkdir(parents=True, exist_ok=True)
    base_name = pdf_path.stem

    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc):
            crop_rect_tuple, strategy = _select_page_crop_rect(page)
            if crop_rect_tuple is None:
                continue

            final_rect = fitz.Rect(*crop_rect_tuple)
            print(
                f"\n>>> 在第 {page_index + 1} 页发现目标区域"
                f" ({strategy}): {final_rect}"
            )

            text_content = page.get_text("text", clip=final_rect).strip()
            output_img_path = output_root / f"{base_name}_page{page_index + 1}.png"
            output_txt_path = output_root / f"{base_name}_page{page_index + 1}.txt"

            pix = page.get_pixmap(clip=final_rect, dpi=target_dpi)
            pix.save(output_img_path)

            print(f"【文字内容提取自第 {page_index + 1} 页】：")
            if text_content:
                print("-" * 30)
                print(text_content)
                print("-" * 30)
                output_txt_path.write_text(text_content, encoding="utf-8")
            else:
                print("(该区域未检测到可直接读取的文字)")

            print(f"【图片已保存】：{output_img_path} ({target_dpi} DPI)")
            return str(output_img_path)

    return None
