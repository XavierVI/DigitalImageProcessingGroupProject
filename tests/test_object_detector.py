import pytest
from unittest.mock import MagicMock
from PIL import Image
import torch

from alert_system.object_detection.object_detector import ObjectDetector


def make_image(width=640, height=480):
    """Create a plain RGB image for testing."""
    return Image.new("RGB", (width, height), color=(120, 120, 120))


def make_mock_post_process_result(scores, labels, boxes):
    """Build the dict that image_processor.post_process_object_detection returns."""
    return [{
        "scores": torch.tensor(scores, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.long),
        "boxes": torch.tensor(boxes, dtype=torch.float32),
    }]


def build_detector(scores, labels, boxes):
    """Return an ObjectDetector wired with mock model outputs."""
    processor = MagicMock()
    model = MagicMock()

    # id2label mapping used inside detect()
    model.config.id2label = {
        0: "car",
        1: "person",
        2: "stop sign",
        3: "traffic light",
        4: "truck",
    }
    model.to = MagicMock(return_value=model)

    # processor(images=...) returns something with a .to() method
    mock_inputs = MagicMock()
    mock_inputs.to = MagicMock(return_value=mock_inputs)
    processor.return_value = mock_inputs

    # post_process_object_detection returns our fake results
    processor.post_process_object_detection.return_value = (
        make_mock_post_process_result(scores, labels, boxes)
    )

    return ObjectDetector(processor, model, device=torch.device("cpu"))


class TestDetectReturnStructure:

    def test_returns_objects_key(self):
        detector = build_detector([0.95], [0], [[10., 20., 100., 200.]])
        result = detector.detect(make_image())
        assert "objects" in result

    def test_returns_raw_results_key(self):
        detector = build_detector([0.95], [0], [[10., 20., 100., 200.]])
        result = detector.detect(make_image())
        assert "raw_results" in result

    def test_objects_is_a_list(self):
        detector = build_detector([0.95], [0], [[10., 20., 100., 200.]])
        result = detector.detect(make_image())
        assert isinstance(result["objects"], list)

    def test_empty_when_no_detections(self):
        detector = build_detector([], [], [])
        result = detector.detect(make_image())
        assert result["objects"] == []

