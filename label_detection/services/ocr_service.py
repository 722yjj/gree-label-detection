import logging
import os
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple

from label_detection.core.langchain_compat import ensure_langchain_legacy_imports
from label_detection.core.config import (
    OCR_BACKEND,
    OCR_DEVICE,
    OCR_LANG,
    OCR_USE_ANGLE_CLS,
    RAPIDOCR_MODEL_TYPE,
    RAPIDOCR_PYTHON,
    RAPIDOCR_TIMEOUT,
    RAPIDOCR_TRT_CACHE_DIR,
    RAPIDOCR_USE_CLS,
)
from label_detection.core.paddle_runtime import resolve_paddle_device

if TYPE_CHECKING:
    from paddleocr import PaddleOCR

# 禁用 PaddleOCR 的大量调试日志
os.environ["PPOCR_KEY_VALUE_CACHE_HEIGHT"] = "False"
logging.getLogger("ppocr").setLevel(logging.ERROR)

# 全局 OCR 引擎实例（懒加载）
_ocr_engine = None

SUPPORTED_RAPIDOCR_BACKENDS = {
    "rapidocr-onnxruntime-cpu",
    "rapidocr-tensorrt",
}


def get_ocr_engine() -> "PaddleOCR":
    """获取或初始化 OCR 引擎（全局单例）"""
    global _ocr_engine
    if _ocr_engine is None:
        ensure_langchain_legacy_imports()
        from paddleocr import PaddleOCR

        device = resolve_paddle_device(OCR_DEVICE, component="PaddleOCR")
        print(f"正在初始化 PaddleOCR (device={device})...")
        _ocr_engine = PaddleOCR(
            use_angle_cls=OCR_USE_ANGLE_CLS,
            lang=OCR_LANG,
            device=device,
        )
    return _ocr_engine


def _resolve_ocr_backend() -> str:
    return os.getenv("OCR_BACKEND", OCR_BACKEND).strip().lower() or "paddleocr"


def get_ocr_backend_name() -> str:
    return _resolve_ocr_backend()


def _resolve_rapidocr_python() -> str:
    return os.getenv("RAPIDOCR_PYTHON", RAPIDOCR_PYTHON).strip() or sys.executable


def _resolve_rapidocr_timeout() -> float:
    value = os.getenv("RAPIDOCR_TIMEOUT", str(RAPIDOCR_TIMEOUT)).strip()
    try:
        return float(value)
    except ValueError:
        return RAPIDOCR_TIMEOUT


def _rapidocr_worker_script() -> str:
    return r'''
import json
import os
import sys

from rapidocr import RapidOCR, EngineType, LangDet, LangRec, ModelType, OCRVersion

image_path = sys.argv[1]
output_path = sys.argv[2]
backend = sys.argv[3]


def env_flag(name, default):
    value = os.getenv(name, default).strip().lower()
    return value not in {"", "0", "false", "no", "off"}

engine_type = {
    "rapidocr-onnxruntime-cpu": EngineType.ONNXRUNTIME,
    "rapidocr-tensorrt": EngineType.TENSORRT,
}[backend]
model_type = {
    "mobile": ModelType.MOBILE,
    "server": ModelType.SERVER,
}.get(os.getenv("RAPIDOCR_MODEL_TYPE", "mobile").strip().lower(), ModelType.MOBILE)
use_cls = env_flag("RAPIDOCR_USE_CLS", "0")

if not use_cls:
    import rapidocr.main as rapidocr_main

    class DisabledTextClassifier:
        def __init__(self, cfg):
            pass

        def __call__(self, img_list):
            from rapidocr.ch_ppocr_cls import TextClsOutput

            if not isinstance(img_list, list):
                img_list = [img_list]
            return TextClsOutput(img_list=img_list, cls_res=None, elapse=0.0)

    rapidocr_main.TextClassifier = DisabledTextClassifier

params = {
    "Global.use_cls": use_cls,
    "Det.engine_type": engine_type,
    "Det.lang_type": LangDet.CH,
    "Det.model_type": model_type,
    "Det.ocr_version": OCRVersion.PPOCRV5,
    "Rec.engine_type": engine_type,
    "Rec.lang_type": LangRec.CH,
    "Rec.model_type": model_type,
    "Rec.ocr_version": OCRVersion.PPOCRV5,
}
if use_cls:
    cls_engine_type = EngineType.ONNXRUNTIME if backend == "rapidocr-tensorrt" else engine_type
    params.update({
        "Cls.engine_type": cls_engine_type,
        "Cls.lang_type": LangRec.CH,
        "Cls.model_type": model_type,
        "Cls.ocr_version": OCRVersion.PPOCRV5,
    })
if backend == "rapidocr-tensorrt":
    params["EngineConfig.tensorrt.cache_dir"] = os.getenv(
        "RAPIDOCR_TRT_CACHE_DIR",
        "/home/jnu/models/ocr/tensorrt-engines/gb10",
    )

ocr = RapidOCR(params=params)
result = ocr(image_path)

boxes = result.boxes if result.boxes is not None else []
txts = result.txts if result.txts is not None else []
scores = result.scores if result.scores is not None else []

payload = {
    "texts": list(txts),
    "scores": [float(score) for score in scores],
    "boxes": [box.tolist() if hasattr(box, "tolist") else box for box in boxes],
    "elapse": result.elapse,
    "elapse_list": result.elapse_list,
}

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False)
'''


def _run_rapidocr_sidecar(image_path: str, backend: str) -> Dict:
    python = _resolve_rapidocr_python()
    timeout = _resolve_rapidocr_timeout()
    env = os.environ.copy()
    env.setdefault("RAPIDOCR_MODEL_TYPE", RAPIDOCR_MODEL_TYPE)
    env.setdefault("RAPIDOCR_TRT_CACHE_DIR", RAPIDOCR_TRT_CACHE_DIR)
    env.setdefault("RAPIDOCR_USE_CLS", "1" if RAPIDOCR_USE_CLS else "0")

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix="rapidocr_",
        delete=False,
    ) as output_file:
        output_path = output_file.name

    try:
        proc = subprocess.run(
            [
                python,
                "-c",
                _rapidocr_worker_script(),
                image_path,
                output_path,
                backend,
            ],
            check=False,
            capture_output=True,
            env=env,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(
                f"RapidOCR sidecar failed with exit code {proc.returncode}: {detail}"
            )

        return json.loads(Path(output_path).read_text(encoding="utf-8"))
    finally:
        try:
            Path(output_path).unlink()
        except FileNotFoundError:
            pass


def _rapidocr_payload_to_text_and_boxes(payload: Dict) -> Tuple[str, List]:
    texts = [str(text) for text in payload.get("texts", [])]
    scores = payload.get("scores", [])
    box_points = payload.get("boxes", [])

    boxes = []
    for idx, poly in enumerate(box_points):
        text = texts[idx] if idx < len(texts) else ""
        score = scores[idx] if idx < len(scores) else 0.0
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0
        boxes.append((poly, text, score))

    return "\n".join(texts), boxes


def _get_rapidocr_with_boxes(image_path: str, backend: str) -> Tuple[str, List]:
    print(
        "正在使用 RapidOCR "
        f"(backend={backend}, python={_resolve_rapidocr_python()})..."
    )
    payload = _run_rapidocr_sidecar(image_path, backend)
    return _rapidocr_payload_to_text_and_boxes(payload)


def get_ocr_text(image_path: str) -> str:
    """
    调用 OCR 提取纯文本

    Args:
        image_path: 图片路径

    Returns:
        拼接的纯文本字符串
    """
    backend = _resolve_ocr_backend()
    if backend in SUPPORTED_RAPIDOCR_BACKENDS:
        text, _ = _get_rapidocr_with_boxes(image_path, backend)
        return text
    if backend != "paddleocr":
        print(f"未知 OCR_BACKEND={backend}，回退 PaddleOCR")

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
    调用 OCR，返回文字内容和边界框坐标

    Args:
        image_path: 图片路径

    Returns:
        text: 拼接的纯文本
        boxes: 边界框列表 [(poly, text, confidence), ...]
    """
    backend = _resolve_ocr_backend()
    if backend in SUPPORTED_RAPIDOCR_BACKENDS:
        try:
            return _get_rapidocr_with_boxes(image_path, backend)
        except Exception as e:
            print(f"RapidOCR 识别出错: {str(e)}")
            return "", []
    if backend != "paddleocr":
        print(f"未知 OCR_BACKEND={backend}，回退 PaddleOCR")

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
