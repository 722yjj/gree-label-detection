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


def test_summarize_result_counts_label_text_difference():
    result = {
        "success": True,
        "verdict": "需复核",
        "text_detection": {
            "fields": ["weight", "label:weight"],
            "template_data": {"weight": "13.5kg", "label:weight": "Weight"},
            "target_data": {"weight": "13.5kg", "label:weight": "weiGht"},
        },
        "graphic_comparison": {
            "resolved_match_count": 0,
            "comparison_results": [],
        },
    }

    summary = DetectionService._summarize_result(result)

    assert summary["text_match_count"] == 1
    assert summary["text_total_count"] == 2


def test_detection_service_passes_output_mode_to_workflow(tmp_path, monkeypatch):
    from desktop_app.models import DetectionJobRequest, TemplateRecord

    monkeypatch.setenv("DESKTOP_DETECTION_PIPELINE", "unified")
    template_path = tmp_path / "template.png"
    target_path = tmp_path / "target.jpg"
    template_path.write_bytes(b"template")
    target_path.write_bytes(b"target")
    captured = {}

    def fake_run(template_input_path, target_image_path, output_dir, output_mode="debug"):
        output_dir_path = tmp_path / "results"
        output_dir_path.mkdir(parents=True, exist_ok=True)
        (output_dir_path / "visualization_diff.jpg").write_bytes(b"vis")
        captured["template_input_path"] = template_input_path
        captured["target_image_path"] = target_image_path
        captured["output_dir"] = output_dir
        captured["output_mode"] = output_mode
        return {
            "success": True,
            "verdict": "需复核",
            "output_dir": str(output_dir_path),
            "text_detection": {
                "fields": ["model"],
                "template_data": {"model": "A"},
                "target_data": {"model": "A"},
            },
            "graphic_comparison": {
                "resolved_match_count": 0,
                "comparison_results": [],
            },
        }

    monkeypatch.setattr(
        "label_detection.workflows.unified.run_unified_detection",
        fake_run,
    )

    request = DetectionJobRequest(
        template=TemplateRecord(
            code="600004075219",
            variant=None,
            display_name="600004075219",
            source_type="image",
            source_path=template_path,
        ),
        target_image_path=target_path,
        output_mode="final",
    )

    service = DetectionService(output_root=tmp_path / "desktop-results")
    result = service.run(request)

    assert captured["template_input_path"] == str(template_path)
    assert captured["target_image_path"] == str(target_path)
    assert captured["output_mode"] == "final"
    assert result.visualization_path == tmp_path / "results" / "visualization_diff.jpg"


def test_detection_service_defaults_to_traditional_full_image_diff(tmp_path, monkeypatch):
    from desktop_app.models import DetectionJobRequest, TemplateRecord

    monkeypatch.setenv("DESKTOP_TRADITIONAL_DIFF_MODEL", "test-model")
    template_path = tmp_path / "template.png"
    target_path = tmp_path / "target.jpg"
    template_path.write_bytes(b"template")
    target_path.write_bytes(b"target")
    captured = {}

    def fake_run(
        template,
        target,
        *,
        output_dir,
        run_name,
        output_mode,
        **kwargs,
    ):
        case_dir = tmp_path / "desktop-results" / "600004075219" / run_name
        case_dir.mkdir(parents=True, exist_ok=True)
        final_image = case_dir / "final_result.jpg"
        result_json = case_dir / "result.json"
        final_image.write_bytes(b"image")
        result_json.write_text("{}", encoding="utf-8")
        captured["template"] = template
        captured["target"] = target
        captured["output_dir"] = output_dir
        captured["run_name"] = run_name
        captured["output_mode"] = output_mode
        captured["kwargs"] = kwargs
        return {
            "success": True,
            "output_dir": str(case_dir),
            "final_box_count": 1,
            "raw_candidate_box_count": 2,
            "merged_candidate_box_count": 1,
            "final_boxes": [{"box": [1, 2, 3, 4]}],
            "vlm_filter_results": [{"vlm_decision": "keep"}],
            "artifacts": {"final_result": str(final_image)},
        }

    monkeypatch.setattr(
        "label_detection.workflows.traditional_full_image_diff.run_traditional_full_image_diff",
        fake_run,
    )

    request = DetectionJobRequest(
        template=TemplateRecord(
            code="600004075219",
            variant=None,
            display_name="600004075219",
            source_type="image",
            source_path=template_path,
        ),
        target_image_path=target_path,
        output_mode="final",
    )

    service = DetectionService(output_root=tmp_path / "desktop-results")
    result = service.run(request)

    assert captured["template"] == template_path
    assert captured["target"] == target_path
    assert captured["output_mode"] == "final"
    assert result.verdict == "不一致"
    assert result.graphic_mismatch_count == 1
    assert result.visualization_path == result.output_dir / "visualization_diff.jpg"
    assert sorted(path.name for path in result.output_dir.iterdir()) == [
        "final_result.json",
        "visualization_diff.jpg",
    ]


def test_traditional_diff_model_resolves_single_vllm_served_name(monkeypatch):
    import desktop_app.services.detection_service as detection_service

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"id": "qwen3.6-27b-int4"}]}

    monkeypatch.setattr(
        detection_service,
        "GRAPHIC_VLM_MODEL",
        "/home/jnu/models/Qwen3.6-27B-int4-AutoRound",
    )
    monkeypatch.delenv("DESKTOP_TRADITIONAL_DIFF_MODEL", raising=False)
    monkeypatch.setattr(
        detection_service.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(),
    )

    assert (
        detection_service._resolve_traditional_diff_model(
            "http://127.0.0.1:8000/v1",
            "EMPTY",
        )
        == "qwen3.6-27b-int4"
    )


def test_traditional_diff_model_keeps_explicit_desktop_model(monkeypatch):
    import desktop_app.services.detection_service as detection_service

    monkeypatch.setenv("DESKTOP_TRADITIONAL_DIFF_MODEL", "manual-model")

    assert (
        detection_service._resolve_traditional_diff_model(
            "http://127.0.0.1:8000/v1",
            "EMPTY",
        )
        == "manual-model"
    )
