from label_detection.services import ocr_service


def test_rapidocr_payload_to_text_and_boxes():
    payload = {
        "texts": ["Model", "GWH18AAD"],
        "scores": [0.98, "0.91"],
        "boxes": [
            [[0, 0], [10, 0], [10, 5], [0, 5]],
            [[12, 0], [30, 0], [30, 5], [12, 5]],
        ],
    }

    text, boxes = ocr_service._rapidocr_payload_to_text_and_boxes(payload)

    assert text == "Model\nGWH18AAD"
    assert boxes == [
        ([[0, 0], [10, 0], [10, 5], [0, 5]], "Model", 0.98),
        ([[12, 0], [30, 0], [30, 5], [12, 5]], "GWH18AAD", 0.91),
    ]


def test_get_ocr_with_boxes_dispatches_to_rapidocr_sidecar(monkeypatch):
    calls = []

    def fake_sidecar(image_path, backend):
        calls.append((image_path, backend))
        return {
            "texts": ["GREE"],
            "scores": [0.99],
            "boxes": [[[1, 2], [3, 2], [3, 4], [1, 4]]],
        }

    monkeypatch.setenv("OCR_BACKEND", "rapidocr-onnxruntime-cpu")
    monkeypatch.setattr(ocr_service, "_run_rapidocr_sidecar", fake_sidecar)

    text, boxes = ocr_service.get_ocr_with_boxes("/tmp/sample.jpg")

    assert calls == [("/tmp/sample.jpg", "rapidocr-onnxruntime-cpu")]
    assert text == "GREE"
    assert boxes == [([[1, 2], [3, 2], [3, 4], [1, 4]], "GREE", 0.99)]


def test_rapidocr_sidecar_disables_cls_by_default(monkeypatch):
    captured = {}

    def fake_run(args, **kwargs):
        captured["env"] = kwargs["env"]
        output_path = args[4]
        with open(output_path, "w", encoding="utf-8") as f:
            f.write('{"texts": [], "scores": [], "boxes": []}')

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.delenv("RAPIDOCR_USE_CLS", raising=False)
    monkeypatch.setattr(ocr_service.subprocess, "run", fake_run)

    ocr_service._run_rapidocr_sidecar("/tmp/sample.jpg", "rapidocr-tensorrt")

    assert captured["env"]["RAPIDOCR_USE_CLS"] == "0"
