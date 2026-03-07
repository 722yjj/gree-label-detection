"""
透视矫正模块

提供标签四角点检测和透视变换矫正功能。

STATUS: main
"""

import cv2
import numpy as np
import os


def order_points(pts):
    """
    将四个角点按顺序排列：左上、右上、右下、左下
    """
    rect = np.zeros((4, 2), dtype=np.float32)

    # 左上点的 x+y 最小，右下点的 x+y 最大
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上
    rect[2] = pts[np.argmax(s)]  # 右下

    # 右上点的 y-x 最小，左下点的 y-x 最大
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上
    rect[3] = pts[np.argmax(diff)]  # 左下

    return rect


def detect_and_correct_perspective(target, template, output_dir=None):
    """
    检测标签四角点并进行透视矫正

    算法：
    1. 检测标签区域轮廓（黑框或浅色区域）
    2. 多边形近似得到四角点
    3. 透视变换矫正为水平矩形

    Args:
        target: 目标图片 (BGR)
        template: 模板图片 (BGR)，用于确定目标尺寸
        output_dir: 输出目录

    Returns:
        corrected: 矫正后的图片
        success: 是否成功检测到四角点
    """
    print("\n[透视矫正] 检测标签四角点...")

    h, w = target.shape[:2]
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    corners = None
    method_used = None

    # ===== 方法1: 检测黑色边框 =====
    print("  尝试方法1: 黑框检测...")
    _, thresh_black = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    thresh_black = cv2.morphologyEx(thresh_black, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        thresh_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        area_ratio = area / (w * h)

        if 0.3 < area_ratio < 0.99:
            epsilon = 0.02 * cv2.arcLength(largest, True)
            approx = cv2.approxPolyDP(largest, epsilon, True)

            if len(approx) == 4:
                corners = approx.reshape(4, 2)
                method_used = "黑框检测"
                print(f"    ✓ 检测到四角点 (占比: {area_ratio:.2%})")

    # ===== 方法2: 检测浅色标签区域 =====
    if corners is None:
        print("  尝试方法2: 浅色区域检测...")
        hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)

        lower = np.array([0, 0, 150])
        upper = np.array([180, 80, 255])
        mask = cv2.inRange(hsv, lower, upper)

        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            area_ratio = area / (w * h)

            if 0.3 < area_ratio < 0.99:
                epsilon = 0.02 * cv2.arcLength(largest, True)
                approx = cv2.approxPolyDP(largest, epsilon, True)

                if len(approx) == 4:
                    corners = approx.reshape(4, 2)
                    method_used = "浅色区域检测"
                    print(f"    ✓ 检测到四角点 (占比: {area_ratio:.2%})")

    # ===== 方法3: 边缘检测 =====
    if corners is None:
        print("  尝试方法3: 边缘检测...")
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 30, 100)

        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            area_ratio = area / (w * h)

            if 0.3 < area_ratio < 0.95:
                epsilon = 0.02 * cv2.arcLength(largest, True)
                approx = cv2.approxPolyDP(largest, epsilon, True)

                if len(approx) == 4:
                    corners = approx.reshape(4, 2)
                    method_used = "边缘检测"
                    print(f"    ✓ 检测到四角点 (占比: {area_ratio:.2%})")

    # 如果没有检测到四角点
    if corners is None:
        print("  ⚠ 未能检测到四角点，跳过透视矫正")
        return target, False

    print(f"  使用方法: {method_used}")

    # 排序角点
    ordered_corners = order_points(corners.astype(np.float32))

    # 计算目标矩形尺寸
    width_top = np.linalg.norm(ordered_corners[1] - ordered_corners[0])
    width_bottom = np.linalg.norm(ordered_corners[2] - ordered_corners[3])
    height_left = np.linalg.norm(ordered_corners[3] - ordered_corners[0])
    height_right = np.linalg.norm(ordered_corners[2] - ordered_corners[1])

    max_width = int(max(width_top, width_bottom))
    max_height = int(max(height_left, height_right))

    # 目标点（水平矩形）
    dst_points = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )

    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(ordered_corners, dst_points)

    # 应用透视变换
    corrected = cv2.warpPerspective(
        target,
        M,
        (max_width, max_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    print(f"  ✓ 透视矫正完成，输出尺寸: {corrected.shape[:2]}")

    # 保存中间结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

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
