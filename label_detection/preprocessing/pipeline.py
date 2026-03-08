"""
预处理流水线模块

提供模板和目标图片的完整预处理流程。

STATUS: main
"""

import cv2
import numpy as np
import os

from .border import crop_to_border, find_black_border
from .perspective import detect_and_correct_perspective


def preprocess_template(template_path, output_dir="results/preprocessed"):
    """
    预处理模板图片：检测并裁剪黑框内区域

    Args:
        template_path: 模板图片路径
        output_dir: 输出目录

    Returns:
        (cropped_image, border_rect, error_msg)
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n[预处理] 处理模板图片...")
    template = cv2.imread(template_path)

    if template is None:
        return None, None, f"无法读取图片: {template_path}"

    print(f"  原始尺寸: {template.shape}")

    # 检测黑框
    border = find_black_border(template, output_dir, "template")

    if border is None:
        print("  未检测到明确的黑框，使用原图")
        return template, None, None

    # 裁剪
    cropped = crop_to_border(template, border, padding=3)
    print(f"  裁剪后尺寸: {cropped.shape}")

    # 保存预处理结果
    output_path = os.path.join(output_dir, "template_cropped.jpg")
    cv2.imwrite(output_path, cropped)
    print(f"  已保存: {output_path}")

    return cropped, border, None


def preprocess_target(target_path, output_dir="results/preprocessed", template_image=None):
    """
    预处理目标图片：检测标签四角点并进行透视矫正

    策略：
    0. 如果提供模板图片，先尝试四角点透视矫正（一步完成矫正+裁剪）
    1. 如果透视矫正失败，回退到黑框检测和边缘检测裁剪逻辑

    Args:
        target_path: 目标图片路径
        output_dir: 输出目录
        template_image: 模板图片（用于透视矫正参考）

    Returns:
        (cropped_image, border_rect, error_msg)
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n[预处理] 处理目标图片...")
    target = cv2.imread(target_path)

    if target is None:
        return None, None, f"无法读取图片: {target_path}"

    print(f"  原始尺寸: {target.shape}")

    # ===== 透视矫正（如果提供了模板图片） =====
    if template_image is not None:
        corrected, success = detect_and_correct_perspective(target, template_image, output_dir)
        if success:
            print("  透视矫正成功，直接返回矫正结果")
            output_path = os.path.join(output_dir, "target_cropped.jpg")
            cv2.imwrite(output_path, corrected)
            return corrected, None, None
        else:
            print("  透视矫正失败，回退到黑框检测和边缘检测...")

    ih, iw = target.shape[:2]

    border = None
    method_used = None

    # ===== 方法1: 黑框检测 =====
    print("  尝试方法1: 黑框检测...")
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    _, thresh_black = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((5, 5), np.uint8)
    thresh_black = cv2.morphologyEx(thresh_black, cv2.MORPH_CLOSE, kernel)

    if output_dir:
        cv2.imwrite(os.path.join(output_dir, "target_thresh.jpg"), thresh_black)

    contours, _ = cv2.findContours(thresh_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        area_ratio = (w * h) / (iw * ih)

        if 0.3 < area_ratio < 0.99:
            border = (x, y, w, h)
            method_used = "黑框检测"
            print(f"    ✓ 检测到黑框: x={x}, y={y}, w={w}, h={h} (占比: {area_ratio:.2%})")

    # ===== 方法3: 边缘检测 =====
    if border is None:
        print("  尝试方法3: 边缘检测...")
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 30, 100)

        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)

        if output_dir:
            cv2.imwrite(os.path.join(output_dir, "target_edges.jpg"), edges)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            area_ratio = (w * h) / (iw * ih)

            if 0.3 < area_ratio < 0.95:
                border = (x, y, w, h)
                method_used = "边缘检测"
                print(f"    ✓ 检测到边界: x={x}, y={y}, w={w}, h={h} (占比: {area_ratio:.2%})")

    # ===== 裁剪 =====
    if border is None:
        print("  ⚠ 所有方法均未检测到有效边界，使用原图")
        return target, None, None

    print(f"  使用方法: {method_used}")
    cropped = crop_to_border(target, border, padding=5)
    print(f"  裁剪后尺寸: {cropped.shape}")

    # 保存
    output_path = os.path.join(output_dir, "target_cropped.jpg")
    cv2.imwrite(output_path, cropped)
    print(f"  已保存: {output_path}")

    return cropped, border, None
