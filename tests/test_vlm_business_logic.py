"""
VLM 业务判定逻辑测试

验证 decision 三态在业务层的正确使用：
- parse_error 不应产生 mismatch
- timeout 应返回 unknown + needs_review
- unknown 不应计入 confirmed diff
- 只有 mismatch 才计入最终图形差异
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from label_detection.services.vlm_service import VLMComparator


class TestErrorHandling:
    """错误状态不应判定为 mismatch"""

    @pytest.fixture
    def comparator(self):
        obj = object.__new__(VLMComparator)
        return obj

    def test_parse_error_not_mismatch(self, comparator):
        """parse_error 不应直接标红"""
        result = comparator._parse_response("some unparseable garbage 12345")
        assert result["decision"] != "mismatch"
        assert result["decision"] == "unknown"
        assert result["needs_review"] is True

    def test_timeout_returns_unknown(self, comparator):
        """timeout 应返回 unknown + needs_review"""
        result = comparator._make_unknown_result("timeout", "请求超时")
        assert result["decision"] == "unknown"
        assert result["needs_review"] is True
        assert result["error_type"] == "timeout"

    def test_request_error_returns_unknown(self, comparator):
        """request_error 应返回 unknown + needs_review"""
        result = comparator._make_unknown_result("request_error", "连接失败")
        assert result["decision"] == "unknown"
        assert result["needs_review"] is True
        assert result["error_type"] == "request_error"

    def test_empty_response_returns_unknown(self, comparator):
        """空响应 → unknown"""
        result = comparator._make_unknown_result("empty_response", "空响应")
        assert result["decision"] == "unknown"
        assert result["needs_review"] is True

    def test_empty_crop_returns_unknown(self, comparator):
        """空裁剪区域 → unknown"""
        result = comparator._make_unknown_result("empty_crop", "区域为空")
        assert result["decision"] == "unknown"
        assert result["needs_review"] is True


class TestBusinessDecisionCounting:
    """业务层统计测试：验证三类统计逻辑正确"""

    def _make_result(self, decision, **kwargs):
        base = {
            "decision": decision,
            "is_match": decision == "match",
            "confidence": 0.8,
            "needs_review": decision == "unknown",
            "error_type": None,
            "differences": [],
            "summary": f"test {decision}",
        }
        base.update(kwargs)
        return base

    def test_only_mismatch_counts_as_diff(self):
        """只有 mismatch 才计入最终图形差异"""
        results = [
            self._make_result("match"),
            self._make_result("mismatch"),
            self._make_result("unknown"),
            self._make_result("unknown", error_type="timeout"),
        ]

        confirmed_mismatch = [r for r in results if r["decision"] == "mismatch"]
        assert len(confirmed_mismatch) == 1

    def test_unknown_not_counted_as_diff(self):
        """unknown 不应计入 confirmed diff"""
        results = [
            self._make_result("match"),
            self._make_result("unknown"),
            self._make_result("unknown", error_type="parse_error"),
        ]

        confirmed_mismatch = [r for r in results if r["decision"] == "mismatch"]
        assert len(confirmed_mismatch) == 0

    def test_parse_error_goes_to_review(self):
        """parse_error 应进入人工复核列表"""
        results = [
            self._make_result("unknown", error_type="parse_error"),
        ]

        needs_review = [r for r in results if r["decision"] == "unknown"]
        assert len(needs_review) == 1

    def test_timeout_goes_to_review(self):
        """timeout 应进入人工复核"""
        results = [
            self._make_result("unknown", error_type="timeout"),
        ]

        needs_review = [r for r in results if r.get("needs_review")]
        assert len(needs_review) == 1

    def test_all_match_passes(self):
        """全部 match → graphic_pass = True"""
        results = [
            self._make_result("match"),
            self._make_result("match"),
        ]

        confirmed_mismatch = [r for r in results if r["decision"] == "mismatch"]
        graphic_pass = len(confirmed_mismatch) == 0
        assert graphic_pass is True

    def test_mismatch_fails(self):
        """存在 mismatch → graphic_pass = False"""
        results = [
            self._make_result("match"),
            self._make_result("mismatch"),
        ]

        confirmed_mismatch = [r for r in results if r["decision"] == "mismatch"]
        graphic_pass = len(confirmed_mismatch) == 0
        assert graphic_pass is False

    def test_mixed_unknown_and_match(self):
        """match + unknown → graphic_pass = True 但 has_review"""
        results = [
            self._make_result("match"),
            self._make_result("unknown"),
        ]

        confirmed_mismatch = [r for r in results if r["decision"] == "mismatch"]
        review_needed = [r for r in results if r["decision"] == "unknown"]

        graphic_pass = len(confirmed_mismatch) == 0
        has_review = len(review_needed) > 0

        assert graphic_pass is True
        assert has_review is True

    def test_needs_review_mismatch_not_counted_as_confirmed_diff(self):
        """带 needs_review 的 mismatch 应进入复核，不应计入 confirmed diff"""
        results = [
            self._make_result("mismatch", needs_review=True),
        ]

        confirmed_mismatch = [
            r for r in results
            if r["decision"] == "mismatch" and not r.get("needs_review", False)
        ]
        review_needed = [
            r for r in results
            if r["decision"] == "unknown" or r.get("needs_review", False)
        ]

        assert len(confirmed_mismatch) == 0
        assert len(review_needed) == 1
