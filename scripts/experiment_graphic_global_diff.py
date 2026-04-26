"""Global aligned graphic-difference experiment.

This script is intentionally isolated from the production workflow. It tests
whether a preprocessed template and target can be compared as whole aligned
images, with text/barcode masks and pixel-tolerance filtering to suppress
alignment noise.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from label_detection.core import langchain_compat as _langchain_compat  # noqa: F401
from label_detection.core.config import PROJECT_ROOT as REPO_ROOT
from label_detection.extraction.template_source import resolve_template_input
from label_detection.preprocessing.pipeline import preprocess_target, preprocess_template


DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "graphic_global_diff_experiment"


Box = Tuple[int, int, int, int]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experiment: whole-image graphic diff after coarse alignment."
    )
    parser.add_argument("--template", required=True, help="Template PDF or image path.")
    parser.add_argument("--target", required=True, help="Target image path.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory.",
    )
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="Do not delete the output directory before running.",
    )
    parser.add_argument(
        "--skip-feature-align",
        action="store_true",
        help="Skip SIFT/ORB homography alignment after preprocessing.",
    )
    parser.add_argument(
        "--skip-ecc",
        action="store_true",
        help="Skip ECC residual alignment after feature alignment.",
    )
    parser.add_argument(
        "--ecc-motion",
        choices=("translation", "euclidean", "affine"),
        default="euclidean",
        help="ECC motion model.",
    )
    parser.add_argument(
        "--skip-ocr-mask",
        action="store_true",
        help="Do not run OCR to mask text regions.",
    )
    parser.add_argument(
        "--skip-barcode-mask",
        action="store_true",
        help="Do not attempt barcode-like region masking.",
    )
    parser.add_argument(
        "--barcode-pad-x",
        type=int,
        default=10,
        help="Horizontal padding around barcode-like masks.",
    )
    parser.add_argument(
        "--barcode-pad-y",
        type=int,
        default=28,
        help="Vertical padding around barcode-like masks.",
    )
    parser.add_argument(
        "--ocr-pad",
        type=int,
        default=6,
        help="Padding around OCR text boxes when masking.",
    )
    parser.add_argument(
        "--diff-tolerance",
        type=int,
        default=4,
        help="Pixel tolerance for residual alignment/line-width differences.",
    )
    parser.add_argument(
        "--merge-radius",
        type=int,
        default=8,
        help="Morphological radius used to merge nearby diff pixels.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=120,
        help="Minimum connected diff component area.",
    )
    parser.add_argument(
        "--min-side",
        type=int,
        default=8,
        help="Minimum width and height for a reported diff box.",
    )
    parser.add_argument(
        "--min-density",
        type=float,
        default=0.025,
        help="Minimum foreground density inside a diff component box.",
    )
    parser.add_argument(
        "--ignore-edge-margin",
        type=int,
        default=10,
        help="Drop diff components touching the comparison-canvas edge.",
    )
    parser.add_argument(
        "--max-boxes",
        type=int,
        default=80,
        help="Maximum boxes retained in the summary.",
    )
    return parser


def prepare_output_dir(path: Path, keep_output: bool) -> None:
    if path.exists() and not keep_output:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def imwrite(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write image: {path}")


def resolve_and_preprocess(
    template_path: str,
    target_path: str,
    output_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    source_dir = output_dir / "template_source"
    preprocess_dir = output_dir / "preprocess"
    source_dir.mkdir(parents=True, exist_ok=True)
    preprocess_dir.mkdir(parents=True, exist_ok=True)

    resolved_template, source_type, error = resolve_template_input(
        template_path,
        output_dir=str(source_dir),
    )
    if error or resolved_template is None:
        raise RuntimeError(error or "Failed to resolve template input.")

    template_img, _, error = preprocess_template(
        resolved_template,
        output_dir=str(preprocess_dir),
    )
    if error or template_img is None:
        raise RuntimeError(error or "Failed to preprocess template.")

    target_img, _, error = preprocess_target(
        target_path,
        output_dir=str(preprocess_dir),
        template_image=template_img,
    )
    if error or target_img is None:
        raise RuntimeError(error or "Failed to preprocess target.")

    return template_img, target_img, {
        "template_input": str(template_path),
        "template_source_type": source_type,
        "resolved_template": str(resolved_template),
        "target_input": str(target_path),
        "preprocess_dir": str(preprocess_dir),
    }


def resize_to_reference(moving: np.ndarray, reference_shape: Sequence[int]) -> np.ndarray:
    height, width = int(reference_shape[0]), int(reference_shape[1])
    if moving.shape[:2] == (height, width):
        return moving.copy()
    return cv2.resize(moving, (width, height), interpolation=cv2.INTER_AREA)


def _make_feature_detector() -> Tuple[Any, str]:
    try:
        return cv2.SIFT_create(nfeatures=2500), "sift"
    except Exception:
        return cv2.ORB_create(nfeatures=2500), "orb"


def feature_align(reference: np.ndarray, moving: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    gray_mov = cv2.cvtColor(moving, cv2.COLOR_BGR2GRAY)
    detector, detector_name = _make_feature_detector()

    kp_ref, des_ref = detector.detectAndCompute(gray_ref, None)
    kp_mov, des_mov = detector.detectAndCompute(gray_mov, None)
    stats: Dict[str, Any] = {
        "method": detector_name,
        "reference_keypoints": len(kp_ref),
        "moving_keypoints": len(kp_mov),
        "good_matches": 0,
        "success": False,
    }

    if des_ref is None or des_mov is None or len(kp_ref) < 4 or len(kp_mov) < 4:
        stats["reason"] = "not_enough_keypoints"
        return moving.copy(), stats

    norm = cv2.NORM_HAMMING if detector_name == "orb" else cv2.NORM_L2
    matcher = cv2.BFMatcher(norm)
    try:
        matches = matcher.knnMatch(des_mov, des_ref, k=2)
    except cv2.error as exc:
        stats["reason"] = f"match_failed: {exc}"
        return moving.copy(), stats

    good = []
    for item in matches:
        if len(item) != 2:
            continue
        first, second = item
        if first.distance < 0.75 * second.distance:
            good.append(first)
    stats["good_matches"] = len(good)

    if len(good) < 8:
        stats["reason"] = "not_enough_good_matches"
        return moving.copy(), stats

    src = np.float32([kp_mov[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    homography, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if homography is None:
        stats["reason"] = "homography_failed"
        return moving.copy(), stats

    inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    stats["inliers"] = inliers
    if inliers < 8:
        stats["reason"] = "not_enough_inliers"
        return moving.copy(), stats

    height, width = reference.shape[:2]
    aligned = cv2.warpPerspective(
        moving,
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    stats["success"] = True
    stats["homography"] = homography.tolist()
    return aligned, stats


def ecc_align(
    reference: np.ndarray,
    moving: np.ndarray,
    motion: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    motion_map = {
        "translation": cv2.MOTION_TRANSLATION,
        "euclidean": cv2.MOTION_EUCLIDEAN,
        "affine": cv2.MOTION_AFFINE,
    }
    motion_type = motion_map[motion]
    warp = np.eye(2, 3, dtype=np.float32)

    gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    gray_mov = cv2.cvtColor(moving, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.equalizeHist(gray_ref).astype(np.float32) / 255.0
    gray_mov = cv2.equalizeHist(gray_mov).astype(np.float32) / 255.0

    stats: Dict[str, Any] = {"motion": motion, "success": False}
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        80,
        1e-5,
    )
    try:
        cc, warp = cv2.findTransformECC(
            gray_ref,
            gray_mov,
            warp,
            motion_type,
            criteria,
            None,
            5,
        )
    except cv2.error as exc:
        stats["reason"] = str(exc)
        return moving.copy(), stats

    height, width = reference.shape[:2]
    aligned = cv2.warpAffine(
        moving,
        warp,
        (width, height),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    stats["success"] = True
    stats["correlation"] = float(cc)
    stats["warp"] = warp.tolist()
    return aligned, stats


def build_foreground_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    block_size = max(35, (min(gray.shape[:2]) // 18) | 1)
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        9,
    )

    mask = cv2.bitwise_and(otsu, adaptive)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    return mask


def expand_box(box: Box, padding: int, shape: Sequence[int]) -> Box:
    height, width = int(shape[0]), int(shape[1])
    x1, y1, x2, y2 = box
    return (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(width, x2 + padding),
        min(height, y2 + padding),
    )


def poly_to_box(poly: Any) -> Box | None:
    points = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    if points.size == 0:
        return None
    x, y, w, h = cv2.boundingRect(points.astype(np.int32))
    return int(x), int(y), int(x + w), int(y + h)


def mask_boxes(mask: np.ndarray, boxes: Iterable[Box], padding: int = 0) -> np.ndarray:
    masked = mask.copy()
    for box in boxes:
        x1, y1, x2, y2 = expand_box(box, padding, masked.shape)
        if x2 > x1 and y2 > y1:
            masked[y1:y2, x1:x2] = 0
    return masked


def mask_boxes_xy(
    mask: np.ndarray,
    boxes: Iterable[Box],
    pad_x: int = 0,
    pad_y: int = 0,
) -> np.ndarray:
    masked = mask.copy()
    height, width = masked.shape[:2]
    for x1, y1, x2, y2 in boxes:
        xx1 = max(0, x1 - pad_x)
        yy1 = max(0, y1 - pad_y)
        xx2 = min(width, x2 + pad_x)
        yy2 = min(height, y2 + pad_y)
        if xx2 > xx1 and yy2 > yy1:
            masked[yy1:yy2, xx1:xx2] = 0
    return masked


def get_ocr_boxes_for_mask(image_path: Path, padding: int) -> Tuple[List[Box], Dict[str, Any]]:
    try:
        from label_detection.services.ocr_service import get_ocr_with_boxes
    except Exception as exc:
        return [], {"success": False, "reason": f"import_failed: {exc}"}

    try:
        _, ocr_boxes = get_ocr_with_boxes(str(image_path))
    except Exception as exc:
        return [], {"success": False, "reason": f"ocr_failed: {exc}"}

    boxes: List[Box] = []
    for poly, text, confidence in ocr_boxes:
        box = poly_to_box(poly)
        if box is None:
            continue
        if not str(text).strip():
            continue
        if float(confidence or 0.0) < 0.35:
            continue
        boxes.append(box)

    return boxes, {
        "success": True,
        "box_count": len(boxes),
        "padding": padding,
    }


def barcode_like_boxes(mask: np.ndarray, image: np.ndarray) -> List[Box]:
    height, width = mask.shape[:2]
    kernel_w = max(17, width // 35)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 3))
    merged = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    merged = cv2.dilate(merged, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)))
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    boxes: List[Box] = []
    image_area = float(height * width)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 70 or h < 18:
            continue
        area = w * h
        aspect = w / max(1.0, float(h))
        if aspect < 2.8 or area / image_area > 0.25:
            continue

        roi = gray[y : y + h, x : x + w]
        _, roi_binary = cv2.threshold(
            roi,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
        column_dark = (roi_binary > 0).mean(axis=0)
        transition_density = float(np.mean(np.abs(np.diff(column_dark)) > 0.08))
        sobel_x = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
        vertical_bias = float(np.mean(np.abs(sobel_x)) / (np.mean(np.abs(sobel_y)) + 1e-6))

        if transition_density >= 0.12 and vertical_bias >= 1.35:
            boxes.append((x, y, x + w, y + h))
    return boxes


def tolerant_diff(
    template_mask: np.ndarray,
    target_mask: np.ndarray,
    tolerance: int,
    merge_radius: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if tolerance > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (tolerance * 2 + 1, tolerance * 2 + 1),
        )
        template_dilated = cv2.dilate(template_mask, kernel)
        target_dilated = cv2.dilate(target_mask, kernel)
    else:
        template_dilated = template_mask
        target_dilated = target_mask

    missing = cv2.bitwise_and(template_mask, cv2.bitwise_not(target_dilated))
    extra = cv2.bitwise_and(target_mask, cv2.bitwise_not(template_dilated))
    diff = cv2.bitwise_or(missing, extra)

    if merge_radius > 0:
        merge_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (merge_radius * 2 + 1, merge_radius * 2 + 1),
        )
        diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, merge_kernel)
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, clean_kernel)
    return missing, extra, diff


def extract_boxes(
    diff_mask: np.ndarray,
    min_area: int,
    min_side: int,
    min_density: float,
    ignore_edge_margin: int,
    max_boxes: int,
) -> List[Dict[str, Any]]:
    image_h, image_w = diff_mask.shape[:2]
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(diff_mask, 8)
    components: List[Dict[str, Any]] = []
    for idx in range(1, count):
        x, y, w, h, area = [int(v) for v in stats[idx]]
        if ignore_edge_margin > 0 and (
            x <= ignore_edge_margin
            or y <= ignore_edge_margin
            or x + w >= image_w - ignore_edge_margin
            or y + h >= image_h - ignore_edge_margin
        ):
            continue
        if area < min_area:
            continue
        if w < min_side or h < min_side:
            continue
        density = area / max(1.0, float(w * h))
        if density < min_density:
            continue
        cx, cy = centroids[idx]
        components.append(
            {
                "box": [x, y, x + w, y + h],
                "area": area,
                "density": float(density),
                "centroid": [float(cx), float(cy)],
            }
        )

    components.sort(key=lambda item: int(item["area"]), reverse=True)
    return components[:max_boxes]


def draw_boxes(image: np.ndarray, components: Sequence[Dict[str, Any]]) -> np.ndarray:
    canvas = image.copy()
    for index, component in enumerate(components, start=1):
        x1, y1, x2, y2 = [int(v) for v in component["box"]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(
            canvas,
            f"D{index}",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
        )
    return canvas


def make_overlay(target: np.ndarray, missing: np.ndarray, extra: np.ndarray) -> np.ndarray:
    overlay = target.copy()
    red = np.zeros_like(target)
    red[:, :] = (0, 0, 255)
    blue = np.zeros_like(target)
    blue[:, :] = (255, 0, 0)
    overlay = np.where(missing[:, :, None] > 0, (0.55 * overlay + 0.45 * red).astype(np.uint8), overlay)
    overlay = np.where(extra[:, :, None] > 0, (0.55 * overlay + 0.45 * blue).astype(np.uint8), overlay)
    return overlay


def make_panel(images: Sequence[Tuple[str, np.ndarray]]) -> np.ndarray:
    max_height = max(image.shape[0] for _, image in images)
    resized = []
    for label, image in images:
        scale = max_height / image.shape[0]
        item = cv2.resize(
            image,
            (int(round(image.shape[1] * scale)), max_height),
            interpolation=cv2.INTER_AREA,
        )
        cv2.putText(
            item,
            label,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        resized.append(item)
    separator = np.full((max_height, 12, 3), 255, dtype=np.uint8)
    parts: List[np.ndarray] = []
    for index, image in enumerate(resized):
        if index:
            parts.append(separator)
        parts.append(image)
    return np.hstack(parts)


def main() -> int:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    prepare_output_dir(output_dir, args.keep_output)

    template_img, target_img, summary = resolve_and_preprocess(
        args.template,
        args.target,
        output_dir,
    )

    template_compare = template_img.copy()
    target_resized = resize_to_reference(target_img, template_compare.shape[:2])
    imwrite(output_dir / "template_compare.jpg", template_compare)
    imwrite(output_dir / "target_resized.jpg", target_resized)

    aligned = target_resized.copy()
    alignment: Dict[str, Any] = {
        "target_original_shape": list(target_img.shape[:2]),
        "comparison_shape": list(template_compare.shape[:2]),
    }
    if not args.skip_feature_align:
        aligned, feature_stats = feature_align(template_compare, aligned)
        alignment["feature"] = feature_stats
    if not args.skip_ecc:
        aligned, ecc_stats = ecc_align(template_compare, aligned, args.ecc_motion)
        alignment["ecc"] = ecc_stats
    imwrite(output_dir / "target_aligned.jpg", aligned)

    template_mask = build_foreground_mask(template_compare)
    target_mask = build_foreground_mask(aligned)
    mask_summary: Dict[str, Any] = {}

    if not args.skip_ocr_mask:
        template_ocr_boxes, template_ocr_stats = get_ocr_boxes_for_mask(
            output_dir / "template_compare.jpg",
            args.ocr_pad,
        )
        target_ocr_boxes, target_ocr_stats = get_ocr_boxes_for_mask(
            output_dir / "target_aligned.jpg",
            args.ocr_pad,
        )
        template_mask = mask_boxes(template_mask, template_ocr_boxes, args.ocr_pad)
        target_mask = mask_boxes(target_mask, target_ocr_boxes, args.ocr_pad)
        mask_summary["template_ocr"] = template_ocr_stats
        mask_summary["target_ocr"] = target_ocr_stats

    if not args.skip_barcode_mask:
        template_barcode_boxes = barcode_like_boxes(template_mask, template_compare)
        target_barcode_boxes = barcode_like_boxes(target_mask, aligned)
        template_mask = mask_boxes_xy(
            template_mask,
            template_barcode_boxes,
            pad_x=args.barcode_pad_x,
            pad_y=args.barcode_pad_y,
        )
        target_mask = mask_boxes_xy(
            target_mask,
            target_barcode_boxes,
            pad_x=args.barcode_pad_x,
            pad_y=args.barcode_pad_y,
        )
        mask_summary["template_barcode_boxes"] = [list(box) for box in template_barcode_boxes]
        mask_summary["target_barcode_boxes"] = [list(box) for box in target_barcode_boxes]

    imwrite(output_dir / "template_graphic_mask.png", template_mask)
    imwrite(output_dir / "target_graphic_mask.png", target_mask)

    missing, extra, diff_mask = tolerant_diff(
        template_mask,
        target_mask,
        tolerance=args.diff_tolerance,
        merge_radius=args.merge_radius,
    )
    components = extract_boxes(
        diff_mask,
        min_area=args.min_area,
        min_side=args.min_side,
        min_density=args.min_density,
        ignore_edge_margin=args.ignore_edge_margin,
        max_boxes=args.max_boxes,
    )

    imwrite(output_dir / "missing_from_target_mask.png", missing)
    imwrite(output_dir / "extra_in_target_mask.png", extra)
    imwrite(output_dir / "diff_mask.png", diff_mask)

    overlay = make_overlay(aligned, missing, extra)
    boxed = draw_boxes(aligned, components)
    imwrite(output_dir / "diff_overlay.jpg", overlay)
    imwrite(output_dir / "target_diff_boxes.jpg", boxed)
    imwrite(
        output_dir / "comparison_panel.jpg",
        make_panel(
            [
                ("template", template_compare),
                ("target aligned", aligned),
                ("diff overlay", overlay),
                ("boxes", boxed),
            ]
        ),
    )

    summary.update(
        {
            "alignment": alignment,
            "masking": mask_summary,
            "parameters": {
                "diff_tolerance": args.diff_tolerance,
                "merge_radius": args.merge_radius,
                "min_area": args.min_area,
                "min_side": args.min_side,
                "min_density": args.min_density,
                "ignore_edge_margin": args.ignore_edge_margin,
                "max_boxes": args.max_boxes,
                "skip_ocr_mask": args.skip_ocr_mask,
                "skip_barcode_mask": args.skip_barcode_mask,
                "barcode_pad_x": args.barcode_pad_x,
                "barcode_pad_y": args.barcode_pad_y,
            },
            "foreground_pixels": {
                "template": int(np.count_nonzero(template_mask)),
                "target": int(np.count_nonzero(target_mask)),
                "missing_from_target": int(np.count_nonzero(missing)),
                "extra_in_target": int(np.count_nonzero(extra)),
                "diff": int(np.count_nonzero(diff_mask)),
            },
            "diff_box_count": len(components),
            "diff_boxes": components,
            "artifacts": {
                "template_compare": str(output_dir / "template_compare.jpg"),
                "target_resized": str(output_dir / "target_resized.jpg"),
                "target_aligned": str(output_dir / "target_aligned.jpg"),
                "template_graphic_mask": str(output_dir / "template_graphic_mask.png"),
                "target_graphic_mask": str(output_dir / "target_graphic_mask.png"),
                "diff_mask": str(output_dir / "diff_mask.png"),
                "diff_overlay": str(output_dir / "diff_overlay.jpg"),
                "target_diff_boxes": str(output_dir / "target_diff_boxes.jpg"),
                "comparison_panel": str(output_dir / "comparison_panel.jpg"),
            },
        }
    )
    write_json(output_dir / "summary.json", summary)

    print(f"Output: {output_dir}")
    print(f"Diff boxes: {len(components)}")
    print(f"Panel: {output_dir / 'comparison_panel.jpg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
