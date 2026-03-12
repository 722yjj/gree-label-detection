import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from label_detection.extraction import template_source


class TestResolveTemplateInput:
    def test_returns_original_path_for_image_input(self, tmp_path):
        image_path = tmp_path / "template.PNG"
        image_path.write_bytes(b"fake-image")

        resolved_path, source_type, err = template_source.resolve_template_input(
            str(image_path)
        )

        assert err is None
        assert source_type == "image"
        assert resolved_path == str(image_path)

    def test_delegates_pdf_input_to_pdf_extractor(self, tmp_path, monkeypatch):
        pdf_path = tmp_path / "template.PDF"
        pdf_path.write_bytes(b"%PDF-1.4")
        expected_image_path = tmp_path / "template_page1.png"

        def fake_extract_red_box_info(path, target_dpi=300, output_dir=None):
            assert path == str(pdf_path)
            assert target_dpi == 240
            assert output_dir == str(tmp_path / "assets")
            return str(expected_image_path)

        monkeypatch.setattr(
            template_source,
            "extract_red_box_info",
            fake_extract_red_box_info,
        )

        resolved_path, source_type, err = template_source.resolve_template_input(
            str(pdf_path),
            output_dir=str(tmp_path / "assets"),
            target_dpi=240,
        )

        assert err is None
        assert source_type == "pdf"
        assert resolved_path == str(expected_image_path)

    def test_returns_error_when_pdf_extraction_fails(self, tmp_path, monkeypatch):
        pdf_path = tmp_path / "template.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        monkeypatch.setattr(
            template_source,
            "extract_red_box_info",
            lambda *args, **kwargs: None,
        )

        resolved_path, source_type, err = template_source.resolve_template_input(
            str(pdf_path)
        )

        assert resolved_path is None
        assert source_type == "pdf"
        assert err == "无法从 PDF 提取模板图片"

    def test_returns_error_for_missing_template(self, tmp_path):
        missing_path = tmp_path / "missing.jpg"

        resolved_path, source_type, err = template_source.resolve_template_input(
            str(missing_path)
        )

        assert resolved_path is None
        assert source_type is None
        assert err == f"找不到模板文件: {missing_path}"

    def test_returns_error_for_unsupported_template_type(self, tmp_path):
        template_path = tmp_path / "template.txt"
        template_path.write_text("not supported", encoding="utf-8")

        resolved_path, source_type, err = template_source.resolve_template_input(
            str(template_path)
        )

        assert resolved_path is None
        assert source_type is None
        assert "不支持的模板文件类型" in err
        assert ".txt" in err


class TestSupportedTemplateImages:
    def test_recognizes_supported_image_suffix(self):
        assert template_source.is_supported_template_image("template.jpeg") is True
        assert template_source.is_supported_template_image("template.WEBP") is True

    def test_rejects_non_image_suffix(self):
        assert template_source.is_supported_template_image("template.pdf") is False
        assert template_source.is_supported_template_image("template") is False
