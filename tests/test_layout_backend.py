from label_detection.matching import layout


class DummyHFLayoutWorker:
    def __init__(self):
        self.calls = []

    def predict(self, image_path, threshold):
        self.calls.append((image_path, threshold))
        return [
            {
                "label": "image",
                "score": 0.9,
                "coordinate": [1.0, 2.0, 3.0, 4.0],
            }
        ]


def test_detect_layout_regions_uses_hf_worker(monkeypatch):
    worker = DummyHFLayoutWorker()
    monkeypatch.setenv("LAYOUT_BACKEND", "hf-pytorch-gpu")
    monkeypatch.setattr(layout, "_get_hf_layout_worker", lambda: worker)

    regions = layout.detect_layout_regions("sample.png", threshold=0.42)

    assert regions == [
        {
            "label": "image",
            "score": 0.9,
            "coordinate": [1.0, 2.0, 3.0, 4.0],
        }
    ]
    assert worker.calls == [("sample.png", 0.42)]


def test_get_layout_backend_name_reads_environment(monkeypatch):
    monkeypatch.setenv("LAYOUT_BACKEND", "hf-pytorch-gpu")

    assert layout.get_layout_backend_name() == "hf-pytorch-gpu"
