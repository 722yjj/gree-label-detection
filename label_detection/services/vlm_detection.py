"""Standalone VLM object detection helper for manual capability testing."""

from __future__ import annotations

import base64
import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

try:
    import cv2
except ImportError:  # pragma: no cover - exercised only in minimal test environments
    cv2 = None

from label_detection.core.config import (
    OLLAMA_API_BASE,
    OLLAMA_MODEL,
    VLM_NUM_PREDICT,
    VLM_TIMEOUT,
)


def _require_cv2():
    if cv2 is None:
        raise ImportError("OpenCV 未安装，无法执行 VLM 目标检测脚本")
    return cv2


class VLMObjectDetector:
    """Use a VLM through Ollama to test prompt-based object detection."""

    PROMPT_TEMPLATE = """你是视觉目标检测器。
请检测图中与以下描述匹配的目标：
{query}

图像尺寸信息：
- width: {width}
- height: {height}

输出要求：
1. 只输出一个 JSON 对象，不要输出任何解释、分析、Markdown、代码块或思考过程。
2. JSON 结构固定为：
{{"objects":[{{"label":"目标名","confidence":0.0,"bbox_1000":[x1,y1,x2,y2]}}],"summary":"..."}}
3. `bbox_1000` 使用相对于整张图像的 0-1000 归一化坐标，格式必须是 [x1, y1, x2, y2]。
4. x1 < x2，y1 < y2。
5. 最多返回 {max_objects} 个最确定的目标。
6. 如果没有检测到目标，返回 {{"objects":[],"summary":"no target found"}}。
7. 不确定时不要猜测。
"""

    def __init__(
        self,
        model_name: str | None = None,
        api_base: str | None = None,
        timeout: int | None = None,
        check_model: bool = True,
    ):
        self.model_name = model_name or OLLAMA_MODEL
        self.api_base = (api_base or OLLAMA_API_BASE).rstrip("/")
        self.timeout = timeout or VLM_TIMEOUT
        if check_model:
            self._check_model()

    def _check_model(self) -> bool:
        """Check whether the requested Ollama model is available."""
        try:
            response = requests.get(f"{self.api_base}/api/tags", timeout=5)
            response.raise_for_status()
            models = response.json().get("models", [])
            names = [item.get("name", "") for item in models]
            if any(self.model_name in name for name in names):
                print(f"[VLM-DET] Ollama 模型就绪: {self.model_name}")
                return True

            print(f"[VLM-DET] 警告: 模型 {self.model_name} 未找到")
            print(f"[VLM-DET] 可用模型: {names}")
            print(f"[VLM-DET] 请运行: ollama pull {self.model_name}")
            return False
        except requests.exceptions.ConnectionError:
            print(f"[VLM-DET] 警告: 无法连接到 Ollama ({self.api_base})")
            print("[VLM-DET] 请确保 Ollama 正在运行: ollama serve")
            return False
        except requests.exceptions.RequestException as exc:
            print(f"[VLM-DET] 警告: 检查模型失败: {exc}")
            return False

    def _image_to_base64(self, image: Any) -> str:
        cv2_lib = _require_cv2()
        ok, buffer = cv2_lib.imencode(".jpg", image, [cv2_lib.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise ValueError("图像编码失败，无法发送给 VLM")
        return base64.b64encode(buffer).decode("utf-8")

    def build_prompt(
        self,
        query: str,
        image_shape: Tuple[int, int],
        max_objects: int = 10,
    ) -> str:
        height, width = image_shape
        return self.PROMPT_TEMPLATE.format(
            query=query,
            width=width,
            height=height,
            max_objects=max_objects,
        )

    def detect_objects(
        self,
        image: Any,
        query: str,
        max_objects: int = 10,
        custom_prompt: str | None = None,
    ) -> Dict[str, Any]:
        """Run prompt-based object detection on a single image."""
        if image is None or image.size == 0:
            raise ValueError("输入图像为空")

        image_b64 = self._image_to_base64(image)
        prompt = custom_prompt or self.build_prompt(query, image.shape[:2], max_objects=max_objects)

        messages = [
            {
                "role": "user",
                "content": prompt,
                "images": [image_b64],
            }
        ]

        try:
            response = requests.post(
                f"{self.api_base}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "stream": False,
                    "think": False,
                    "options": {
                        "temperature": 0,
                        "num_predict": VLM_NUM_PREDICT,
                    },
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.exceptions.Timeout as exc:
            raise RuntimeError("VLM 请求超时") from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"VLM 请求失败: {exc}") from exc

        payload = response.json()
        message = payload.get("message", {})
        content = message.get("content", "") or message.get("thinking", "")
        if not content:
            raise RuntimeError("VLM 返回空响应")

        fallback_label = self._build_fallback_label(query)
        parsed = self._parse_response(content, image.shape[:2], fallback_label)
        parsed["raw_response"] = content
        parsed["model"] = self.model_name
        parsed["query"] = query
        parsed["image_size"] = {
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
        }
        return parsed

    def _build_fallback_label(self, query: str) -> str:
        label = str(query or "object").split(",")[0].split("，")[0].strip()
        return label or "object"

    def _parse_response(
        self,
        response: str,
        image_shape: Tuple[int, int],
        default_label: str = "object",
    ) -> Dict[str, Any]:
        payload = self._extract_json_object(response)
        if payload is None:
            return {
                "success": False,
                "parse_error": True,
                "summary": "模型输出解析失败",
                "objects": [],
            }

        raw_objects = self._extract_object_list(payload)
        objects: List[Dict[str, Any]] = []
        for item in raw_objects:
            normalized = self._normalize_detection(item, image_shape, default_label)
            if normalized is not None:
                objects.append(normalized)

        summary = str(
            payload.get("summary")
            or payload.get("message")
            or payload.get("result")
            or ""
        )

        return {
            "success": True,
            "parse_error": False,
            "summary": summary,
            "objects": objects,
        }

    def _extract_json_object(self, response: str) -> Optional[Dict[str, Any]]:
        text = str(response or "").strip()
        if not text:
            return None

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
        if fenced:
            try:
                parsed = json.loads(fenced.group(1))
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", text):
            candidate = text[match.start():]
            try:
                parsed, _ = decoder.raw_decode(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed

        return None

    def _extract_object_list(self, payload: Dict[str, Any]) -> List[Any]:
        for key in ("objects", "regions", "detections", "results", "items", "boxes"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        return []

    def _normalize_detection(
        self,
        item: Any,
        image_shape: Tuple[int, int],
        default_label: str = "object",
    ) -> Optional[Dict[str, Any]]:
        label = default_label
        confidence = 0.0
        bbox = None
        coord_hint = None

        if isinstance(item, dict):
            label = str(
                item.get("label")
                or item.get("name")
                or item.get("category")
                or item.get("class")
                or default_label
            ).strip() or default_label
            confidence = self._safe_float(
                item.get("confidence", item.get("score", item.get("probability", 0.0))),
                default=0.0,
            )
            bbox, coord_hint = self._extract_bbox(item)
        elif isinstance(item, (list, tuple)) and len(item) >= 4:
            bbox = [self._safe_float(value, default=0.0) for value in item[:4]]

        if bbox is None:
            return None

        bbox_px, bbox_1000, coord_mode = self._convert_bbox(
            bbox,
            image_shape,
            coord_hint=coord_hint,
        )
        if bbox_px is None:
            return None

        return {
            "label": label,
            "confidence": confidence,
            "bbox_px": bbox_px,
            "bbox_1000": bbox_1000,
            "coord_mode": coord_mode,
        }

    def _extract_bbox(self, item: Dict[str, Any]) -> Tuple[Optional[List[float]], Optional[str]]:
        candidate_keys = (
            "bbox_1000",
            "bbox",
            "box",
            "coordinates",
            "xyxy",
        )
        for key in candidate_keys:
            if key in item:
                values = self._coerce_bbox(item[key])
                if values is not None:
                    if key == "bbox_1000":
                        return values, "norm_1000"
                    return values, None

        if all(key in item for key in ("x1", "y1", "x2", "y2")):
            return [
                self._safe_float(item["x1"], 0.0),
                self._safe_float(item["y1"], 0.0),
                self._safe_float(item["x2"], 0.0),
                self._safe_float(item["y2"], 0.0),
            ], "pixel"

        if all(key in item for key in ("left", "top", "right", "bottom")):
            return [
                self._safe_float(item["left"], 0.0),
                self._safe_float(item["top"], 0.0),
                self._safe_float(item["right"], 0.0),
                self._safe_float(item["bottom"], 0.0),
            ], "pixel"

        return None, None

    def _coerce_bbox(self, raw_bbox: Any) -> Optional[List[float]]:
        if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) >= 4:
            return [self._safe_float(value, 0.0) for value in raw_bbox[:4]]

        if isinstance(raw_bbox, dict):
            if all(key in raw_bbox for key in ("x1", "y1", "x2", "y2")):
                return [
                    self._safe_float(raw_bbox["x1"], 0.0),
                    self._safe_float(raw_bbox["y1"], 0.0),
                    self._safe_float(raw_bbox["x2"], 0.0),
                    self._safe_float(raw_bbox["y2"], 0.0),
                ]
            if all(key in raw_bbox for key in ("left", "top", "right", "bottom")):
                return [
                    self._safe_float(raw_bbox["left"], 0.0),
                    self._safe_float(raw_bbox["top"], 0.0),
                    self._safe_float(raw_bbox["right"], 0.0),
                    self._safe_float(raw_bbox["bottom"], 0.0),
                ]

        return None

    def _convert_bbox(
        self,
        bbox: Sequence[float],
        image_shape: Tuple[int, int],
        coord_hint: str | None = None,
    ) -> Tuple[Optional[List[int]], Optional[List[int]], str]:
        height, width = image_shape
        if width <= 0 or height <= 0:
            return None, None, "invalid"

        values = [float(value) for value in bbox[:4]]
        max_abs = max(abs(value) for value in values)
        x_values = values[0::2]
        y_values = values[1::2]

        if coord_hint == "ratio":
            coord_mode = "ratio"
        elif coord_hint == "norm_1000":
            coord_mode = "norm_1000"
        elif coord_hint == "pixel":
            coord_mode = "pixel"
        elif max_abs <= 1.5:
            coord_mode = "ratio"
        elif (
            max_abs <= 1001.0
            and (max(x_values, default=0.0) > width or max(y_values, default=0.0) > height)
        ):
            coord_mode = "norm_1000"
        else:
            coord_mode = "pixel"

        if coord_mode == "ratio":
            x1, y1, x2, y2 = (
                values[0] * width,
                values[1] * height,
                values[2] * width,
                values[3] * height,
            )
        elif coord_mode == "norm_1000":
            x1, y1, x2, y2 = (
                values[0] / 1000.0 * width,
                values[1] / 1000.0 * height,
                values[2] / 1000.0 * width,
                values[3] / 1000.0 * height,
            )
        else:
            x1, y1, x2, y2 = values

        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))

        left = int(round(max(0.0, min(left, width - 1))))
        top = int(round(max(0.0, min(top, height - 1))))
        right = int(round(max(1.0, min(right, width))))
        bottom = int(round(max(1.0, min(bottom, height))))

        if right <= left:
            right = min(width, left + 1)
        if bottom <= top:
            bottom = min(height, top + 1)

        bbox_px = [left, top, right, bottom]
        bbox_1000 = [
            int(round(left / width * 1000)),
            int(round(top / height * 1000)),
            int(round(right / width * 1000)),
            int(round(bottom / height * 1000)),
        ]
        return bbox_px, bbox_1000, coord_mode

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default


def draw_detections(
    image: Any,
    detections: Sequence[Dict[str, Any]],
    min_confidence: float = 0.0,
) -> Any:
    """Draw detection boxes on the image for quick manual review."""
    cv2_lib = _require_cv2()
    canvas = image.copy()
    palette = [
        (0, 255, 0),
        (0, 165, 255),
        (255, 0, 0),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]

    for idx, item in enumerate(detections):
        confidence = float(item.get("confidence", 0.0))
        if confidence < min_confidence:
            continue

        x1, y1, x2, y2 = [int(value) for value in item["bbox_px"]]
        color = palette[idx % len(palette)]
        label = str(item.get("label") or "object")
        label_ascii = label.encode("ascii", errors="ignore").decode("ascii") or label
        label_text = f"{idx + 1}:{label_ascii} {confidence:.2f}"

        cv2_lib.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        text_y = y1 - 8 if y1 > 24 else y1 + 20
        cv2_lib.putText(
            canvas,
            label_text,
            (x1, text_y),
            cv2_lib.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

    return canvas


__all__ = ["VLMObjectDetector", "draw_detections"]
