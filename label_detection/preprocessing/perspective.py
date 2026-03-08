"""
透视矫正模块

提供标签四角点检测和透视变换矫正功能。

STATUS: main
"""

import os

import cv2
import numpy as np


def order_points(pts):
    """
    将四个角点按顺序排列：左上、右上、右下、左下
    """
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def _template_aspect_ratio(template):
    """返回模板宽高比，用于过滤明显错误的候选区域。"""
    if template is None or template.size == 0:
        return None

    th, tw = template.shape[:2]
    if th == 0 or tw == 0:
        return None

    return tw / float(th)


def _contour_to_quad(contour):
    """将轮廓尽量转换成四边形。"""
    if contour is None or len(contour) < 4:
        return None

    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return None

    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    if len(approx) == 4:
        return approx.reshape(4, 2).astype(np.float32)

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect).astype(np.float32)
    if cv2.contourArea(box) <= 0:
        return None

    return box


def _score_quad(quad, contour_area, image_shape, template_ratio):
    """
    对候选四边形打分。

    评分目标：
    1. 尽量接近模板长宽比
    2. 轮廓在四边形中填充充分，避免只取到零散文字
    3. 避免直接选中整张图边缘
    """
    ih, iw = image_shape[:2]
    image_area = float(iw * ih)

    quad = order_points(quad.astype(np.float32))
    quad_area = abs(cv2.contourArea(quad))
    if quad_area <= 0:
        return None

    area_ratio = quad_area / image_area
    if area_ratio < 0.03 or area_ratio > 0.98:
        return None

    width_top = np.linalg.norm(quad[1] - quad[0])
    width_bottom = np.linalg.norm(quad[2] - quad[3])
    height_left = np.linalg.norm(quad[3] - quad[0])
    height_right = np.linalg.norm(quad[2] - quad[1])

    max_width = max(width_top, width_bottom)
    max_height = max(height_left, height_right)
    if max_width < 30 or max_height < 30:
        return None

    aspect_ratio = max_width / max(max_height, 1.0)
    if aspect_ratio < 1.0 or aspect_ratio > 12.0:
        return None

    fill_ratio = contour_area / quad_area
    if fill_ratio < 0.35:
        return None

    if template_ratio is not None:
        aspect_gap = abs(np.log(aspect_ratio / template_ratio))
        if aspect_gap > np.log(2.8):
            return None
        aspect_score = max(0.0, 1.4 - aspect_gap)
    else:
        aspect_score = 0.8

    x, y, w, h = cv2.boundingRect(quad.astype(np.int32))
    touch_count = (
        int(x <= 5)
        + int(y <= 5)
        + int(x + w >= iw - 5)
        + int(y + h >= ih - 5)
    )
    edge_penalty = {0: 0.0, 1: 0.1, 2: 0.4, 3: 1.0, 4: 1.5}.get(touch_count, 1.5)

    area_score = 1.0 - min(abs(area_ratio - 0.35) / 0.35, 1.0)
    fill_score = min(fill_ratio, 1.0)
    score = aspect_score * 2.0 + area_score + fill_score - edge_penalty

    return {
        "corners": quad,
        "score": score,
        "area_ratio": area_ratio,
        "aspect_ratio": aspect_ratio,
    }


def _find_best_candidate(mask, image_shape, template_ratio, method_name):
    """从当前掩码中挑选最像标签主体的四边形。"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    image_area = float(image_shape[0] * image_shape[1])
    best = None

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
        contour_area = cv2.contourArea(contour)
        if contour_area / image_area < 0.003:
            continue

        quad = _contour_to_quad(contour)
        if quad is None:
            continue

        candidate = _score_quad(quad, contour_area, image_shape, template_ratio)
        if candidate is None:
            continue

        candidate["method"] = method_name
        if best is None or candidate["score"] > best["score"]:
            best = candidate

    return best


def detect_and_correct_perspective(target, template, output_dir=None):
    """
    检测标签四角点并进行透视矫正。

    当前策略：
    1. 方法1：聚合深色前景（文字、边框、条码），找出最像标签主体的四边形
    2. 方法3：边缘检测后提取候选四边形
    3. 结合模板长宽比对候选区域打分，避免直接选中整张图或外部背景

    Args:
        target: 目标图片 (BGR)
        template: 模板图片 (BGR)，用于提供标签长宽比参考
        output_dir: 输出目录

    Returns:
        corrected: 矫正后的图片
        success: 是否成功检测到四角点
    """
    print("\n[透视矫正] 检测标签四角点...")

    h, w = target.shape[:2]
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    template_ratio = _template_aspect_ratio(template)

    candidates = []

    # ===== 方法1: 聚合深色前景 =====
    print("  尝试方法1: 深色区域聚合...")
    _, dark_mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(15, w // 35), max(15, h // 35))
    )
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, close_kernel)
    dark_mask = cv2.dilate(
        dark_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, w // 180), max(3, h // 180))),
        iterations=1,
    )

    candidate = _find_best_candidate(
        dark_mask, target.shape, template_ratio, "深色区域聚合"
    )
    if candidate is not None:
        candidates.append(candidate)
        print(
            f"    ✓ 找到候选四边形 (占比: {candidate['area_ratio']:.2%}, "
            f"得分: {candidate['score']:.2f})"
        )

    # ===== 方法3: 边缘检测 =====
    print("  尝试方法3: 边缘检测...")
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edge_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(7, w // 80), max(7, h // 80))
    )
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, edge_kernel)
    edges = cv2.dilate(edges, edge_kernel, iterations=1)

    candidate = _find_best_candidate(edges, target.shape, template_ratio, "边缘检测")
    if candidate is not None:
        candidates.append(candidate)
        print(
            f"    ✓ 找到候选四边形 (占比: {candidate['area_ratio']:.2%}, "
            f"得分: {candidate['score']:.2f})"
        )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, "target_method1_mask.jpg"), dark_mask)
        cv2.imwrite(os.path.join(output_dir, "target_method3_edges.jpg"), edges)

    if not candidates:
        print("  ⚠ 未能检测到四角点，跳过透视矫正")
        return target, False

    best = max(candidates, key=lambda item: item["score"])
    ordered_corners = best["corners"]
    method_used = best["method"]
    print(f"  使用方法: {method_used}")

    width_top = np.linalg.norm(ordered_corners[1] - ordered_corners[0])
    width_bottom = np.linalg.norm(ordered_corners[2] - ordered_corners[3])
    height_left = np.linalg.norm(ordered_corners[3] - ordered_corners[0])
    height_right = np.linalg.norm(ordered_corners[2] - ordered_corners[1])

    max_width = max(1, int(round(max(width_top, width_bottom))))
    max_height = max(1, int(round(max(height_left, height_right))))

    if template_ratio is not None:
        detected_ratio = max_width / float(max_height)
        if detected_ratio >= template_ratio:
            max_height = max(1, int(round(max_width / template_ratio)))
        else:
            max_width = max(1, int(round(max_height * template_ratio)))

    dst_points = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(ordered_corners, dst_points)
    corrected = cv2.warpPerspective(
        target,
        M,
        (max_width, max_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    print(f"  ✓ 透视矫正完成，输出尺寸: {corrected.shape[:2]}")

    if output_dir:
        debug_img = target.copy()
        for i, pt in enumerate(ordered_corners):
            cv2.circle(debug_img, tuple(pt.astype(int)), 10, (0, 0, 255), -1)
            cv2.putText(
                debug_img,
                str(i),
                tuple(pt.astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2,
            )
        cv2.polylines(
            debug_img, [ordered_corners.astype(np.int32)], True, (0, 255, 0), 3
        )
        cv2.imwrite(
            os.path.join(output_dir, "target_corners_detected.jpg"), debug_img
        )
        cv2.imwrite(
            os.path.join(output_dir, "target_perspective_corrected.jpg"), corrected
        )

    return corrected, True
