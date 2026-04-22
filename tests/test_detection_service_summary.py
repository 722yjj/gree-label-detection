from desktop_app.services.detection_service import DetectionService


def test_summarize_result_extracts_text_and_graphic_counts():
    result = {
        "success": True,
        "verdict": "需复核",
        "text_detection": {
            "fields": ["model", "barcode"],
            "template_data": {"model": "A", "barcode": "123"},
            "target_data": {"model": "A", "barcode": "456"},
        },
        "graphic_comparison": {
            "resolved_match_count": 3,
            "comparison_results": [
                {"decision": "match"},
                {"decision": "mismatch"},
                {"decision": "unknown"},
            ],
        },
    }

    summary = DetectionService._summarize_result(result)

    assert summary == {
        "text_match_count": 1,
        "text_total_count": 2,
        "graphic_match_count": 3,
        "graphic_mismatch_count": 1,
        "graphic_review_count": 1,
    }


def test_build_summary_text_uses_structured_counts():
    result = {"success": True, "verdict": "需复核"}
    summary = {
        "text_match_count": 14,
        "text_total_count": 14,
        "graphic_match_count": 5,
        "graphic_mismatch_count": 1,
        "graphic_review_count": 0,
    }

    text = DetectionService._build_summary_text(result, summary)

    assert "综合判定: 需复核" in text
    assert "文字字段匹配: 14/14" in text
    assert "图形匹配: 5" in text
    assert "图形不一致: 1" in text
