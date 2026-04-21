"""
统一 VLM (视觉大模型) 服务模块

封装 Ollama API 调用 Qwen3-VL 的逻辑，提供图形对比和结构化提取能力。

返回结果包含 decision 三态字段 (match / mismatch / unknown)，
业务层应优先使用 decision 而非 is_match 进行判定。

STATUS: main
"""

import json
import base64
import re
import requests
import numpy as np
from typing import Dict, Optional

try:
    import cv2
except ImportError:  # pragma: no cover - exercised only in minimal test environments
    cv2 = None

from label_detection.core.config import (
    OLLAMA_API_BASE,
    OLLAMA_MODEL,
    VLM_TIMEOUT,
    VLM_NUM_PREDICT,
    VLM_CANVAS_SIZE,
    VLM_MAX_RETRIES,
)


def _require_cv2():
    if cv2 is None:
        raise ImportError("OpenCV 未安装，无法执行图像对比")
    return cv2


class VLMComparator:
# class VLMComparator:
    """使用 Qwen3-VL (Ollama) 进行图形对比的类"""

    # Original Chinese prompt kept for reference:
    # 你是图形一致性判定器。
    # 你将看到左右两张局部图形区域图：左图是 Template，右图是 Target。
    #
    # 判定目标：
    # 只比较前景图标/符号本身是否一致。
    # 不要把以下因素视为差异：
    # - 裁剪位置偏移
    # - 留白差异
    # - 轻微模糊
    # - 亮度或颜色变化
    # - 小于15%的缩放差异
    # - 条形码、二维码、条码下方数字或纯编码区域
    #
    # 重点检查：
    # 1. 图标数量是否一致
    # 2. 是否有缺失或多余图标
    # 3. 图标主体形状是否明显不同
    #
    # 输出规则（必须严格遵守）：
    # 1. 只输出一个 JSON 对象，不要输出任何其他文本。
    # 2. 禁止输出思考过程、解释、分析、推理、注释、Markdown、代码块、标签（例如 <think>）。
    # 3. JSON 必须且仅包含以下 4 个键：
    #    - "decision": "match" | "mismatch" | "unknown"
    #    - "confidence": 0.0 到 1.0 的数字
    #    - "differences": 字符串数组
    #    - "summary": 字符串
    # 4. 如果区域主体是条形码、二维码、条码数字或纯编码，直接输出 "decision":"match"，并将 "summary" 固定为 "barcode_ignored"。
    # 5. 如果无法可靠判断，必须输出 "decision": "unknown"，不要猜测。
    # 6. 当 decision 为 "match" 时，differences 必须是空数组 []。
    #
    # 仅输出如下格式的 JSON：
    # {"decision":"match|mismatch|unknown","confidence":0.0,"differences":[],"summary":"..."}
    PROMPT_TEMPLATE = """You are a graphic consistency judge.
You will see two local graphic regions side by side:
- left: Template
- right: Target

Task:
Compare only the foreground icons / symbols themselves.
Do NOT treat the following as differences:
- crop offset
- whitespace difference
- slight blur
- brightness or color change
- scaling difference smaller than 15%
- barcode, QR code, barcode digits, or pure code regions

Focus on:
1. whether the number of icons is the same
2. whether any icon is missing or extra
3. whether the main icon shapes are clearly different

Output rules (must follow strictly):
1. Output exactly one JSON object and nothing else.
2. Do not output thinking, explanation, analysis, reasoning, comments, Markdown, code fences, or tags such as <think>.
3. The JSON must contain exactly these 4 keys:
   - "decision": "match" | "mismatch" | "unknown"
   - "confidence": a number from 0.0 to 1.0
   - "differences": an array of strings
   - "summary": a string
4. If the region is mainly a barcode, QR code, barcode digits, or pure code, output "decision":"match" and set "summary" to "barcode_ignored".
5. If you cannot judge reliably, output "decision":"unknown". Do not guess.
6. If decision is "match", then differences must be [].

Output only JSON in this format:
{"decision":"match|mismatch|unknown","confidence":0.0,"differences":[],"summary":"..."}"""

    def __init__(
        self,
        model_name: str = None,
        api_base: str = None,
        timeout: int = None,
    ):
        """
        初始化 VLM 对比器

        Args:
            model_name: Ollama 模型名称（默认从 config 读取）
            api_base: Ollama API 地址（默认从 config 读取）
            timeout: 请求超时时间（秒）
        """
        self.model_name = model_name or OLLAMA_MODEL
        self.api_base = (api_base or OLLAMA_API_BASE).rstrip("/")
        self.timeout = timeout or VLM_TIMEOUT
        self._check_model()

    def _check_model(self) -> bool:
        """检查模型是否可用"""
        try:
            resp = requests.get(f"{self.api_base}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                model_names = [m.get("name", "") for m in models]

                if any(self.model_name in name for name in model_names):
                    print(f"[VLM] Ollama 模型就绪: {self.model_name}")
                    return True
                else:
                    print(f"[VLM] 警告: 模型 {self.model_name} 未找到")
                    print(f"[VLM] 可用模型: {model_names}")
                    print(f"[VLM] 请运行: ollama pull {self.model_name}")
                    return False
        except requests.exceptions.ConnectionError:
            print(f"[VLM] 警告: 无法连接到 Ollama ({self.api_base})")
            print(f"[VLM] 请确保 Ollama 正在运行: ollama serve")
        return False

    def _image_to_base64(self, image: np.ndarray) -> str:
        """将 OpenCV 图像转换为 base64 编码"""
        cv2_lib = _require_cv2()
        _, buffer = cv2_lib.imencode(".jpg", image, [cv2_lib.IMWRITE_JPEG_QUALITY, 95])
        return base64.b64encode(buffer).decode("utf-8")

    def _standardize_crop(self, crop: np.ndarray) -> np.ndarray:
        """
        标准化裁剪区域：前景检测 + 去空白 + 统一尺寸

        Args:
            crop: 裁剪的区域图像 (BGR)

        Returns:
            标准化后的图像 (VLM_CANVAS_SIZE x VLM_CANVAS_SIZE)
        """
        cv2_lib = _require_cv2()
        target_size = VLM_CANVAS_SIZE

        # 前景检测：灰度阈值 + 轮廓包围盒
        gray = cv2_lib.cvtColor(crop, cv2_lib.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        _, binary = cv2_lib.threshold(gray, 240, 255, cv2_lib.THRESH_BINARY_INV)

        contours, _ = cv2_lib.findContours(binary, cv2_lib.RETR_EXTERNAL, cv2_lib.CHAIN_APPROX_SIMPLE)

        if contours:
            # 合并所有轮廓的包围盒
            all_points = np.vstack(contours)
            x, y, w, h = cv2_lib.boundingRect(all_points)
            # 加 padding 避免裁剪过紧
            pad = max(5, int(min(w, h) * 0.05))
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(crop.shape[1] - x, w + 2 * pad)
            h = min(crop.shape[0] - y, h + 2 * pad)
            cropped_fg = crop[y:y+h, x:x+w]
        else:
            cropped_fg = crop

        # 按长边缩放，短边补白
        h, w = cropped_fg.shape[:2]
        scale = target_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2_lib.resize(cropped_fg, (new_w, new_h), interpolation=cv2_lib.INTER_AREA)

        # 创建白色画布并居中放置
        canvas = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255
        x_offset = (target_size - new_w) // 2
        y_offset = (target_size - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

        return canvas

    def _create_comparison_canvas(
        self,
        template_crop: np.ndarray,
        target_crop: np.ndarray,
    ) -> np.ndarray:
        """
        创建标准化拼接画布：左模板 + 中间分隔 + 右实拍

        Args:
            template_crop: 模板区域裁剪
            target_crop: 实拍区域裁剪

        Returns:
            拼接画布 (VLM_CANVAS_SIZE x VLM_CANVAS_SIZE*2+20)
        """
        cv2_lib = _require_cv2()
        size = VLM_CANVAS_SIZE
        sep_width = 20

        std_template = self._standardize_crop(template_crop)
        std_target = self._standardize_crop(target_crop)

        # 拼接画布
        canvas_w = size * 2 + sep_width
        canvas_h = size + 40  # 上方留空给标题
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

        # 分隔线
        sep_x = size
        cv2_lib.line(canvas, (sep_x + sep_width // 2, 0), (sep_x + sep_width // 2, canvas_h), (180, 180, 180), 2)

        # 放置图像
        canvas[40:40+size, 0:size] = std_template
        canvas[40:40+size, size+sep_width:size+sep_width+size] = std_target

        # 添加标题
        cv2_lib.putText(canvas, "Template", (size // 2 - 40, 30), cv2_lib.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2_lib.putText(canvas, "Target", (size + sep_width + size // 2 - 30, 30), cv2_lib.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        return canvas

    def compare_images(
        self,
        template_image: np.ndarray,
        target_image: np.ndarray,
        custom_prompt: str = None,
    ) -> Dict:
        """
        对比两张图片（带重试和 schema 校验）

        Args:
            template_image: 模板图像 (numpy array, BGR 格式)
            target_image: 目标图像 (numpy array, BGR 格式)
            custom_prompt: 自定义提示词（可选）

        Returns:
            对比结果字典，包含 decision 字段
        """
        # 创建标准化拼接画布
        canvas = self._create_comparison_canvas(template_image, target_image)
        canvas_b64 = self._image_to_base64(canvas)

        prompt = custom_prompt or self.PROMPT_TEMPLATE

        max_retries = VLM_MAX_RETRIES
        last_error = None

        for attempt in range(max_retries):
            try:
                messages = [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [canvas_b64],
                    }
                ]

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

                result = response.json()
                message = result.get("message", {})

                content = message.get("content", "")
                if not content:
                    content = message.get("thinking", "")

                if not content:
                    print(f"[VLM] 警告: API 响应为空 (尝试 {attempt+1}/{max_retries})")
                    if attempt < max_retries - 1:
                        continue
                    return self._make_unknown_result(
                        error_type="empty_response",
                        summary="API 返回空响应",
                    )

                parsed = self._parse_response(content)

                # 校验: 如果解析出 unknown 且还有重试机会，再试一次
                if parsed.get("decision") == "unknown" and parsed.get("parse_error") and attempt < max_retries - 1:
                    print(f"[VLM] 解析失败，重试 ({attempt+1}/{max_retries})")
                    continue

                return parsed

            except requests.exceptions.Timeout:
                last_error = "timeout"
                if attempt < max_retries - 1:
                    print(f"[VLM] 请求超时，重试 ({attempt+1}/{max_retries})")
                    continue
                return self._make_unknown_result(
                    error_type="timeout",
                    summary="对比失败：请求超时",
                )
            except requests.exceptions.RequestException as e:
                last_error = str(e)
                if attempt < max_retries - 1:
                    print(f"[VLM] 请求失败，重试 ({attempt+1}/{max_retries}): {e}")
                    continue
                return self._make_unknown_result(
                    error_type="request_error",
                    summary=f"对比失败：{str(e)}",
                )

        # 不应到达这里，但以防万一
        return self._make_unknown_result(
            error_type="max_retries_exceeded",
            summary="对比失败：超过最大重试次数",
        )

    def _make_unknown_result(self, error_type: str, summary: str) -> Dict:
        """构造 unknown 结果"""
        return {
            "decision": "unknown",
            "is_match": False,
            "confidence": 0.0,
            "needs_review": True,
            "error_type": error_type,
            "differences": [],
            "summary": summary,
        }

    def _parse_response(self, response: str) -> Dict:
        """
        解析模型响应为结构化结果

        返回结果始终包含: decision, is_match, confidence, differences, summary, needs_review
        """
        response = str(response or "").strip()

        # 尝试解析 JSON
        try:
            # 优先匹配包含 decision 字段的 JSON
            json_match = re.search(r'\{[^{}]*"decision"[^{}]*\}', response)
            if json_match:
                result = json.loads(json_match.group())
                decision = result.get("decision", "unknown")
                if decision not in ("match", "mismatch", "unknown"):
                    decision = "unknown"
                return self._normalize_result(result, decision)

            # 兼容旧格式: 匹配 is_match 字段
            json_match = re.search(r'\{[^{}]*"is_match"[^{}]*\}', response)
            if json_match:
                result = json.loads(json_match.group())
                is_match = result.get("is_match", False)
                decision = "match" if is_match else "mismatch"
                return self._normalize_result(result, decision)

        except json.JSONDecodeError:
            pass

        if "<think>" in response.lower():
            return {
                "decision": "unknown",
                "is_match": False,
                "confidence": 0.0,
                "needs_review": True,
                "error_type": "parse_error",
                "differences": [],
                "summary": "模型输出了思考内容，但未返回可解析 JSON",
                "raw_response": response[:500],
                "parse_error": True,
            }

        # 关键词兜底推断
        response_lower = response.lower()

        diff_keywords = [
            "不一致", "不同", "缺失", "缺少", "多余", "不匹配",
            "different", "missing", "mismatch", "not match", "inconsistent",
            "差异", "不相同", "有差别",
        ]
        match_keywords = [
            "一致", "相同", "匹配", "完全相同",
            "same", "match", "identical", "consistent",
        ]

        has_diff_keyword = any(kw in response_lower for kw in diff_keywords)
        has_match_keyword = any(kw in response_lower for kw in match_keywords)

        # 差异关键词优先（"不一致" 同时包含 "一致"，但语义是否定的）
        if has_diff_keyword:
            return {
                "decision": "unknown",
                "is_match": False,
                "confidence": 0.0,
                "needs_review": True,
                "error_type": "parse_error",
                "differences": ["从模型文本推断存在差异"],
                "summary": "基于关键词推断可能不匹配（建议复核）",
                "raw_response": response[:500],
                "inferred": True,
                "tentative_decision": "mismatch",
                "parse_error": True,
            }

        if has_match_keyword:
            return {
                "decision": "unknown",
                "is_match": False,
                "confidence": 0.0,
                "needs_review": True,
                "error_type": "parse_error",
                "differences": [],
                "summary": "基于关键词推断可能匹配（建议复核）",
                "raw_response": response[:500],
                "inferred": True,
                "tentative_decision": "match",
                "parse_error": True,
            }

        # 完全无法解析
        return {
            "decision": "unknown",
            "is_match": False,
            "confidence": 0.0,
            "needs_review": True,
            "error_type": "parse_error",
            "differences": [],
            "summary": "模型输出解析失败",
            "raw_response": response[:500],
            "parse_error": True,
        }

    def _normalize_result(self, result: Dict, decision: str) -> Dict:
        """将解析出的 JSON 结果标准化为统一格式"""
        is_match = decision == "match"
        confidence = result.get("confidence", 0.0)
        needs_review = decision == "unknown"

        return {
            "decision": decision,
            "is_match": is_match,
            "confidence": confidence,
            "needs_review": needs_review,
            "error_type": None,
            "differences": result.get("differences", []),
            "summary": result.get("summary", ""),
            "raw_response": str(result)[:500],
        }

    def compare_region_pair(
        self,
        template_img: np.ndarray,
        target_img: np.ndarray,
        template_box: list,
        target_box: list,
        output_dir: str = None,
        pair_idx: int = 0,
    ) -> Dict:
        """
        对比两个区域（与 layout_region_comparison.py 接口兼容）
        """
        import os
        cv2_lib = _require_cv2()

        x1, y1, x2, y2 = [int(v) for v in template_box]
        template_crop = template_img[y1:y2, x1:x2]

        x1, y1, x2, y2 = [int(v) for v in target_box]
        target_crop = target_img[y1:y2, x1:x2]

        if template_crop.size == 0 or target_crop.size == 0:
            return self._make_unknown_result(
                error_type="empty_crop",
                summary="无法对比：区域裁剪为空",
            )

        result = self.compare_images(template_crop, target_crop)

        result["template_box"] = template_box
        result["target_box"] = target_box
        result["template_size"] = list(template_crop.shape[:2])
        result["target_size"] = list(target_crop.shape[:2])

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            cv2_lib.imwrite(
                os.path.join(output_dir, f"vlm_region_{pair_idx}_template.jpg"),
                template_crop,
            )
            cv2_lib.imwrite(
                os.path.join(output_dir, f"vlm_region_{pair_idx}_target.jpg"),
                target_crop,
            )
            # 保存标准化拼接画布供调试
            canvas = self._create_comparison_canvas(template_crop, target_crop)
            cv2_lib.imwrite(
                os.path.join(output_dir, f"vlm_region_{pair_idx}_canvas.jpg"),
                canvas,
            )

        return result


# 全局实例（懒加载）
_vlm_comparator = None


def get_vlm_comparator(
    model_name: str = None,
    api_base: str = None,
) -> VLMComparator:
    """获取或创建 VLM 对比器实例"""
    global _vlm_comparator
    if _vlm_comparator is None:
        _vlm_comparator = VLMComparator(
            model_name=model_name,
            api_base=api_base,
        )
    return _vlm_comparator


def compare_with_vlm(
    template_image: np.ndarray,
    target_image: np.ndarray,
    model_name: str = None,
) -> Dict:
    """
    便捷函数：使用 VLM 对比两张图片
    """
    comparator = get_vlm_comparator(model_name)
    return comparator.compare_images(template_image, target_image)
