"""
边框检测与裁剪模块

提供黑色边框检测和裁剪功能，用于模板图片和实拍图片的预处理。

STATUS: main
"""

import cv2
import numpy as np
import os


def find_black_border(image, debug_dir=None, name="image"):
    """
    检测图片中的黑色边框，返回边框内区域的坐标

    Args:
        image: BGR 图像
        debug_dir: 调试输出目录
        name: 图像名称（用于保存调试图）

    Returns:
        (x, y, w, h): 黑框内区域的边界框
        None: 如果未检测到边框
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 二值化：将黑色区域变白
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    # 形态学操作，连接边框
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, f"{name}_thresh.jpg"), thresh)

    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # 找最大的轮廓（应该是黑框）
    largest = max(contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(largest)

    # 验证：轮廓应该占据图像的大部分
    area_ratio = (cw * ch) / (w * h)
    if area_ratio < 0.3:
        print(f"  警告: 检测到的区域太小 ({area_ratio:.2%})")
        return None

    print(f"  检测到边框区域: x={x}, y={y}, w={cw}, h={ch} (占比: {area_ratio:.2%})")

    return (x, y, cw, ch)


def crop_to_border(image, border_rect, padding=5):
    """
    裁剪图像到边框区域

    Args:
        image: 原始图像
        border_rect: (x, y, w, h) 边框坐标
        padding: 内边距（向内缩进，去除边框线本身）
    """
    x, y, w, h = border_rect
    ih, iw = image.shape[:2]

    # 添加内边距（去除黑色边框线本身）
    x1 = min(x + padding, iw)
    y1 = min(y + padding, ih)
    x2 = max(x + w - padding, 0)
    y2 = max(y + h - padding, 0)

    return image[y1:y2, x1:x2]
