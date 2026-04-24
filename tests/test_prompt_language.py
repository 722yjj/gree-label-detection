from label_detection.schema import LABEL_KIND_COMPACT, LABEL_KIND_STANDARD
from label_detection.services.vlm_detection import VLMObjectDetector
from label_detection.services.vlm_service import VLMComparator
from label_detection.workflows.unified import _build_extraction_prompt


def test_extraction_prompts_are_english_json_instructions():
    standard = _build_extraction_prompt(LABEL_KIND_STANDARD, "OCR TEXT")
    compact = _build_extraction_prompt(LABEL_KIND_COMPACT, "OCR TEXT")

    assert standard.startswith("Task: extract structured text")
    assert compact.startswith("Task: extract structured text")
    assert "Output exactly one valid JSON object" in standard
    assert "Output one valid JSON object only" in compact
    assert "【" not in standard
    assert "【" not in compact


def test_vlm_prompts_are_english_json_instructions():
    comparison_prompt = VLMComparator.PROMPT_TEMPLATE
    detection_prompt = VLMObjectDetector.PROMPT_TEMPLATE

    assert comparison_prompt.startswith("You are a graphic consistency judge.")
    assert detection_prompt.startswith("You are a visual object detector.")
    assert "Output exactly one JSON object" in comparison_prompt
    assert "Output exactly one JSON object" in detection_prompt
    assert "你是" not in comparison_prompt
    assert "你是" not in detection_prompt
