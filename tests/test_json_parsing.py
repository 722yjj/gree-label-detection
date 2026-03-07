"""
VLM 响应 JSON 解析单元测试

测试 VLMComparator._parse_response 对各种格式的处理能力
包含 decision 三态字段 (match / mismatch / unknown) 的测试
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from label_detection.services.vlm_service import VLMComparator


class TestParseResponse:
    """VLM 响应解析测试"""

    @pytest.fixture
    def parser(self):
        """创建解析器（不初始化网络连接）"""
        obj = object.__new__(VLMComparator)
        return obj

    # === JSON 解析 ===

    def test_parse_clean_json_with_decision(self, parser):
        """测试包含 decision 字段的 JSON 响应"""
        response = '{"decision": "match", "confidence": 0.95, "differences": [], "summary": "一致"}'
        result = parser._parse_response(response)
        assert result["decision"] == "match"
        assert result["is_match"] is True
        assert result["confidence"] == 0.95
        assert result["needs_review"] is False

    def test_parse_decision_mismatch(self, parser):
        """测试 decision=mismatch"""
        response = '{"decision": "mismatch", "confidence": 0.8, "differences": ["icon missing"], "summary": "不同"}'
        result = parser._parse_response(response)
        assert result["decision"] == "mismatch"
        assert result["is_match"] is False
        assert result["needs_review"] is False

    def test_parse_decision_unknown(self, parser):
        """测试 decision=unknown 的合法 JSON"""
        response = '{"decision": "unknown", "confidence": 0.3, "differences": [], "summary": "无法判断"}'
        result = parser._parse_response(response)
        assert result["decision"] == "unknown"
        assert result["is_match"] is False
        assert result["needs_review"] is True

    def test_parse_legacy_is_match_true(self, parser):
        """测试旧格式 is_match=true 兼容"""
        response = '{"is_match": true, "confidence": 0.9, "differences": [], "summary": "ok"}'
        result = parser._parse_response(response)
        assert result["decision"] == "match"
        assert result["is_match"] is True

    def test_parse_legacy_is_match_false(self, parser):
        """测试旧格式 is_match=false 兼容"""
        response = '{"is_match": false, "confidence": 0.8, "differences": ["diff"], "summary": "不同"}'
        result = parser._parse_response(response)
        assert result["decision"] == "mismatch"
        assert result["is_match"] is False

    def test_parse_json_with_prefix(self, parser):
        """测试带前缀文本的 JSON"""
        response = '根据分析结果：\n{"decision": "mismatch", "confidence": 0.8, "differences": ["icon missing"], "summary": "不同"}'
        result = parser._parse_response(response)
        assert result["decision"] == "mismatch"

    def test_parse_json_with_codeblock(self, parser):
        """测试代码块中的 JSON"""
        response = '```json\n{"decision": "match", "confidence": 0.9, "differences": [], "summary": "ok"}\n```'
        result = parser._parse_response(response)
        assert "decision" in result

    def test_parse_with_think_tags(self, parser):
        """测试带 <think> 标签的响应"""
        response = '<think>让我分析一下...</think>\n{"decision": "match", "confidence": 0.8, "differences": [], "summary": "相同"}'
        result = parser._parse_response(response)
        assert result["decision"] == "match"

    def test_parse_invalid_decision_value(self, parser):
        """测试无效的 decision 值应回退为 unknown"""
        response = '{"decision": "maybe", "confidence": 0.5, "differences": [], "summary": "不确定"}'
        result = parser._parse_response(response)
        assert result["decision"] == "unknown"
        assert result["needs_review"] is True

    # === 关键词兜底 ===

    def test_parse_diff_keyword_fallback(self, parser):
        """测试关键词兜底 — 不一致"""
        response = "两张图的图标数量不一致，第一张有3个认证标志，第二张缺少CE标志"
        result = parser._parse_response(response)
        assert result["decision"] == "mismatch"
        assert result["is_match"] is False
        assert result.get("inferred") is True
        assert result["needs_review"] is True  # 关键词推断应标记需复核

    def test_parse_match_keyword_fallback(self, parser):
        """测试关键词兜底 — 一致"""
        response = "两张标签的图形内容完全相同，所有图标匹配"
        result = parser._parse_response(response)
        assert result["decision"] == "match"
        assert result["is_match"] is True
        assert result.get("inferred") is True
        assert result["needs_review"] is True  # 关键词推断应标记需复核

    # === 解析失败 ===

    def test_parse_empty_response(self, parser):
        """测试空响应应返回 unknown"""
        result = parser._parse_response("")
        assert result["decision"] == "unknown"
        assert result["is_match"] is False
        assert result["needs_review"] is True
        assert result.get("parse_error") is True

    def test_parse_garbage(self, parser):
        """测试完全不可解析的响应"""
        result = parser._parse_response("asdf1234!@#$")
        assert result["decision"] == "unknown"
        assert result["is_match"] is False
        assert result["needs_review"] is True
        assert result.get("parse_error") is True

    def test_parse_long_text_no_json(self, parser):
        """测试解释性长文本但无 JSON 且无关键词"""
        response = "这是一段很长的分析文本，但是没有包含任何有用的结论性信息。" * 10
        result = parser._parse_response(response)
        assert result["decision"] == "unknown"
        assert result["needs_review"] is True

    # === 结果完整性 ===

    def test_result_always_has_required_keys(self, parser):
        """确保任何输入下结果都包含必需的键"""
        test_inputs = [
            "",
            "random text",
            '{"decision": "match"}',
            "图形一致",
            "不匹配",
            '{"is_match": true}',
        ]
        required_keys = {"decision", "is_match", "confidence", "differences", "summary", "needs_review"}
        for inp in test_inputs:
            result = parser._parse_response(inp)
            for key in required_keys:
                assert key in result, f"Missing key '{key}' for input: {inp!r}"

    def test_decision_and_is_match_consistent(self, parser):
        """decision 和 is_match 应保持一致"""
        cases = [
            ('{"decision": "match", "confidence": 0.9}', True),
            ('{"decision": "mismatch", "confidence": 0.8}', False),
            ('{"decision": "unknown", "confidence": 0.5}', False),
        ]
        for response, expected_is_match in cases:
            result = parser._parse_response(response)
            assert result["is_match"] is expected_is_match, f"For {response}"


class TestMakeUnknownResult:
    """测试 _make_unknown_result 辅助方法"""

    @pytest.fixture
    def comparator(self):
        obj = object.__new__(VLMComparator)
        return obj

    def test_unknown_result_structure(self, comparator):
        result = comparator._make_unknown_result("timeout", "请求超时")
        assert result["decision"] == "unknown"
        assert result["is_match"] is False
        assert result["needs_review"] is True
        assert result["error_type"] == "timeout"
        assert result["confidence"] == 0.0
