"""
文字区域掩盖工具
用于从图片中掩盖 OCR 检测到的文字区域

STATUS: main
"""
import cv2
import numpy as np
import os

# 使用公共 OCR 服务
from services.ocr_service import get_ocr_engine, get_ocr_with_boxes



def mask_text_regions(image, text_boxes, fill_color=(255, 255, 255), padding=2):
    """
    在图片上掩盖指定的文字区域
    
    Args:
        image: BGR 图像 (numpy array) 或图片路径
        text_boxes: 边界框列表 [(poly, text, score), ...]
                   其中 poly 是 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        fill_color: 填充颜色，默认白色 (255, 255, 255)
        padding: 扩展边界的像素数
        
    Returns:
        masked_image: 掩盖后的图像
    """
    # 如果是路径，读取图片
    if isinstance(image, str):
        image = cv2.imread(image)
    
    masked = image.copy()
    
    for box_info in text_boxes:
        poly = box_info[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        
        # 转换为 numpy 数组
        pts = np.array(poly, dtype=np.int32)
        
        # 添加 padding（扩展边界）
        if padding > 0:
            # 计算中心点
            center = pts.mean(axis=0)
            # 向外扩展
            pts = pts + (pts - center) * (padding / np.linalg.norm(pts - center, axis=1, keepdims=True))
            pts = pts.astype(np.int32)
        
        # 填充多边形区域
        cv2.fillPoly(masked, [pts], fill_color)
    
    return masked


def process_image_with_masking(image_path, output_path=None, save_debug=False, debug_dir=None):
    """
    处理图片：OCR 识别 + 掩盖文字区域
    
    Args:
        image_path: 输入图片路径
        output_path: 输出图片路径（可选）
        save_debug: 是否保存调试图片
        debug_dir: 调试图片保存目录
        
    Returns:
        dict: {
            "text": 识别的文本,
            "boxes": 边界框列表,
            "masked_image": 掩盖后的图像,
            "original_image": 原始图像
        }
    """
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        return {"error": f"无法读取图片: {image_path}"}
    
    # OCR 识别
    text, boxes = get_ocr_with_boxes(image_path)
    print(f"  OCR 识别到 {len(boxes)} 个文字区域")
    
    # 掩盖文字区域
    masked_image = mask_text_regions(image, boxes)
    
    # 保存结果
    if output_path:
        cv2.imwrite(output_path, masked_image)
        print(f"  已保存掩盖后图片: {output_path}")
    
    # 保存调试图片（标注边界框）
    if save_debug and debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        debug_image = image.copy()
        for box_info in boxes:
            poly = np.array(box_info[0], dtype=np.int32)
            cv2.polylines(debug_image, [poly], True, (0, 255, 0), 2)
        debug_path = os.path.join(debug_dir, "ocr_boxes.jpg")
        cv2.imwrite(debug_path, debug_image)
        print(f"  已保存调试图片: {debug_path}")
    
    return {
        "text": text,
        "boxes": boxes,
        "masked_image": masked_image,
        "original_image": image
    }


if __name__ == "__main__":
    # 测试
    test_image = "test.jpg"
    if os.path.exists(test_image):
        result = process_image_with_masking(
            test_image, 
            output_path="test_masked.jpg",
            save_debug=True,
            debug_dir="results/debug"
        )
        print(f"\n识别到的文本:\n{result['text'][:200]}...")
    else:
        print(f"测试图片不存在: {test_image}")
