import json
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path

from label_detection.batch_samples import (
    BatchCase,
    SampleAsset,
    build_cases,
    cleanup_report_intermediates,
    compute_text_summary,
    discover_samples,
    extract_label_code,
    run_case,
)


def make_asset(path: Path, code: str) -> SampleAsset:
    return SampleAsset(path=path, code=code, stem=path.stem)


@contextmanager
def workspace_tmp_dir():
    root = Path(__file__).resolve().parent / ".tmp" / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_extract_label_code_supports_variant_suffixes():
    assert extract_label_code("600004075219-01") == "600004075219"
    assert extract_label_code("600004075219_4") == "600004075219"
    assert extract_label_code("prefix_63249929995_1") == "63249929995"
    assert extract_label_code("template") is None


def test_discover_samples_groups_by_shared_code():
    with workspace_tmp_dir() as tmp_path:
        template_root = tmp_path / "templates"
        target_root = tmp_path / "targets"
        template_root.mkdir()
        target_root.mkdir()

        (template_root / "600004075219.pdf").write_text("pdf", encoding="utf-8")
        (template_root / "600004075219-01.pdf").write_text("pdf", encoding="utf-8")
        (target_root / "600004075219_1.jpg").write_text("img", encoding="utf-8")
        (target_root / "not_a_code.jpg").write_text("img", encoding="utf-8")

        discovery = discover_samples([template_root], [target_root])

        assert sorted(discovery.templates_by_code) == ["600004075219"]
        assert sorted(asset.stem for asset in discovery.templates_by_code["600004075219"]) == [
            "600004075219",
            "600004075219-01",
        ]
        assert sorted(discovery.targets_by_code) == ["600004075219"]
        assert len(discovery.ignored_targets) == 1
        assert discovery.ignored_targets[0].endswith("not_a_code.jpg")


def test_build_cases_supports_all_and_best_template_modes():
    with workspace_tmp_dir() as tmp_path:
        code = "600004075219"
        template_root = tmp_path / "templates"
        target_root = tmp_path / "targets"
        template_root.mkdir()
        target_root.mkdir()

        templates = {
            code: [
                make_asset(template_root / "600004075219.pdf", code),
                make_asset(template_root / "600004075219-01.pdf", code),
            ]
        }
        targets = {
            code: [
                make_asset(target_root / "600004075219_1.jpg", code),
                make_asset(target_root / "600004075219_2.jpg", code),
            ]
        }

        all_cases = build_cases(templates, targets, pair_mode="all")
        best_template_cases = build_cases(templates, targets, pair_mode="best-template")

        assert len(all_cases) == 4
        assert len(best_template_cases) == 2
        assert all(case.template.stem == "600004075219" for case in best_template_cases)


def test_run_case_keeps_only_json_and_visualization(monkeypatch):
    with workspace_tmp_dir() as tmp_path:
        code = "600004075219"
        template_path = tmp_path / "templates" / "600004075219.pdf"
        target_path = tmp_path / "targets" / "600004075219_1.jpg"
        template_path.parent.mkdir()
        target_path.parent.mkdir()
        template_path.write_text("pdf", encoding="utf-8")
        target_path.write_text("img", encoding="utf-8")

        case = BatchCase(
            code=code,
            template=make_asset(template_path, code),
            target=make_asset(target_path, code),
            case_id="600004075219__vs__600004075219_1",
        )

        def fake_runner(template: str, target: str, output_dir: str):
            output_path = Path(output_dir)
            (output_path / "visualization_diff.jpg").write_bytes(b"vis")
            (output_path / "template_preprocessed.jpg").write_bytes(b"template")
            (output_path / "target_preprocessed.jpg").write_bytes(b"target")
            (output_path / "text_comparison.xlsx").write_bytes(b"xlsx")
            with (output_path / "final_result.json").open("w", encoding="utf-8") as file:
                json.dump(
                    {
                        "success": True,
                        "verdict": "ok",
                        "text_detection": {
                            "fields": ["barcode", "model_number"],
                            "template_data": {"barcode": code, "model_number": "A"},
                            "target_data": {"barcode": code, "model_number": "B"},
                        },
                        "graphic_comparison": {
                            "layout_backend": "hf-pytorch-gpu",
                            "template_regions_total_count": 2,
                            "target_regions_total_count": 3,
                            "matched_count": 1,
                            "effective_matched_count": 1,
                            "comparison_results": [
                                {"decision": "mismatch", "summary": "diff"}
                            ],
                        },
                    },
                    file,
                    ensure_ascii=False,
                    indent=2,
                )
            return {"success": True}

        monkeypatch.setattr("label_detection.batch_samples.get_detection_runner", lambda: fake_runner)

        output_root = tmp_path / "batch_results"
        record = run_case(case, output_root)
        case_dir = output_root / code / case.case_id

        assert record.success is True
        assert sorted(path.name for path in case_dir.iterdir()) == ["result.json", "visualization_diff.jpg"]

        payload = json.loads((case_dir / "result.json").read_text(encoding="utf-8"))
        assert payload["artifacts"]["visualization_diff"].endswith("visualization_diff.jpg")
        assert payload["text_detection"]["different_fields"] == ["model_number"]
        assert payload["graphic_comparison"]["layout_backend"] == "hf-pytorch-gpu"
        assert payload["graphic_comparison"]["template_regions_total_count"] == 2
        assert payload["graphic_comparison"]["target_regions_total_count"] == 3
        assert payload["graphic_comparison"]["mismatch_count"] == 1
        assert payload["graphic_comparison"]["resolved_match_count"] == 1


def test_compute_text_summary_uses_normalized_field_matching():
    summary = compute_text_summary(
        {
            "text_detection": {
                "fields": ["manufacturer", "mfg_date"],
                "template_data": {
                    "manufacturer": "GREE ELECTRIC APPLIANCES,INC.OF ZHUHAI",
                    "mfg_date": "2026.01",
                },
                "target_data": {
                    "manufacturer": "GREE ELECTRIC APPLIANCES, INC. OF ZHUHAI",
                    "mfg_date": "YYYY.MM",
                },
            }
        }
    )

    assert summary["match_count"] == 1
    assert summary["different_fields"] == ["mfg_date"]


def test_cleanup_report_intermediates_removes_preprocessed_images():
    with workspace_tmp_dir() as tmp_path:
        case_dir = tmp_path / "600004075219" / "case_1"
        case_dir.mkdir(parents=True, exist_ok=True)
        keep_path = case_dir / "result.json"
        keep_path.write_text("{}", encoding="utf-8")
        for name in ("template_preprocessed.jpg", "target_preprocessed.jpg"):
            (case_dir / name).write_bytes(b"img")

        cleanup_report_intermediates(tmp_path)

        assert keep_path.exists()
        assert not (case_dir / "template_preprocessed.jpg").exists()
        assert not (case_dir / "target_preprocessed.jpg").exists()
