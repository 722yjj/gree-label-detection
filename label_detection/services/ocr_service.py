import logging
import os
from typing import TYPE_CHECKING, List, Tuple

from label_detection.core.config import OCR_DEVICE, OCR_LANG, OCR_USE_ANGLE_CLS
from label_detection.core.paddle_runtime import resolve_paddle_device

if TYPE_CHECKING:
    from paddleocr import PaddleOCR

# 禁用 PaddleOCR 的大量调试日志
os.environ["PPOCR_KEY_VALUE_CACHE_HEIGHT"] = "False"
logging.getLogger("ppocr").setLevel(logging.ERROR)

# 全局 OCR 引擎实例（懒加载）
_ocr_engine = None


def get_ocr_engine() -> "PaddleOCR":
    """获取或初始化 OCR 引擎（全局单例）"""
    global _ocr_engine
    if _ocr_engine is None:
        from paddleocr import PaddleOCR

        device = resolve_paddle_device(OCR_DEVICE, component="PaddleOCR")
        print(f"正在初始化 PaddleOCR (device={device})...")
        _ocr_engine = PaddleOCR(
            use_angle_cls=OCR_USE_ANGLE_CLS,
            lang=OCR_LANG,
            device=device,
        )
    return _ocr_engine


def get_ocr_text(image_path: str) -> str:
    """
    调用 PaddleOCR 提取纯文本

    Args:
        image_path: 图片路径

    Returns:
        拼接的纯文本字符串
    """
    ocr_engine = get_ocr_engine()
    try:
        result = ocr_engine.predict(image_path)

        if not result or result[0] is None:
            return ""

        all_texts = []
        for res in result:
            rec_texts = res.get("rec_texts", [])
            all_texts.extend(rec_texts)

        return "\n".join(all_texts)
    except Exception as e:
        print(f"OCR 识别出错: {str(e)}")
        return ""


def get_ocr_with_boxes(image_path: str) -> Tuple[str, List]:
    """
    调用 PaddleOCR，返回文字内容和边界框坐标

    Args:
        image_path: 图片路径

    Returns:
        text: 拼接的纯文本
        boxes: 边界框列表 [(poly, text, confidence), ...]
    """
    ocr_engine = get_ocr_engine()

    try:
        result = ocr_engine.predict(image_path)

        if not result or result[0] is None:
            return "", []

        texts = []
        boxes = []

        for res in result:
            rec_texts = res.get("rec_texts", [])
            rec_scores = res.get("rec_scores", [])
            dt_polys = res.get("dt_polys", [])

            texts.extend(rec_texts)

            for i, poly in enumerate(dt_polys):
                text = rec_texts[i] if i < len(rec_texts) else ""
                score = rec_scores[i] if i < len(rec_scores) else 0.0
                boxes.append((poly, text, score))

        return "\n".join(texts), boxes

    except Exception as e:
        print(f"OCR 识别出错: {str(e)}")
        return "", []
