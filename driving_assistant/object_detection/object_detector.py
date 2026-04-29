"""Object detection using DETR, RT-DETR, or YOLO models."""

import os
from typing import List, Dict, Optional

import numpy as np
import torch
from PIL import Image

from transformers import (
    DetrImageProcessor,
    DetrForObjectDetection,
    RTDetrImageProcessor,
    RTDetrForObjectDetection,
)


try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - optional dependency for YOLO mode
    YOLO = None

class ObjectDetector:
    """Performs object detection on images using pre-trained DETR or YOLO models.

    Args:
        processor: Processor for preparing images for the model
        model: Pre-trained DETR model
        device: Device to run inference on (cuda or cpu)
    """

    def __init__(self, obj_detection_model, device=None, yolo_weights_path: Optional[str] = None):
        """Initialize the object detector.

        Args:
            obj_detection_model: Pre-trained model to load.
                (e.g., "facebook/detr-resnet-50" or "PekingU/rtdetr_r50vd" or "yolo")
            device: torch.device to use for inference
        """
        self.device = device or torch.device("cpu")
        self.model_name = obj_detection_model
        # loading models
        # Load Object Detection Model (DETR)
        model = "PekingU/rtdetr_r50vd"
        # model = "facebook/detr-resnet-50"

        if obj_detection_model == "facebook/detr-resnet-50":
            self.processor = DetrImageProcessor.from_pretrained(
                model,
                cache_dir=os.path.join(os.getcwd(), "models")
            )
            self.obj_det_model = DetrForObjectDetection.from_pretrained(
                model,
                cache_dir=os.path.join(os.getcwd(), "models")
            ).to(self.device)
            self.detection_func = self._detr_detection

        elif obj_detection_model == "PekingU/rtdetr_r50vd":
            self.processor = RTDetrImageProcessor.from_pretrained(
                model,
                cache_dir=os.path.join(os.getcwd(), "models")
            )
            self.obj_det_model = RTDetrForObjectDetection.from_pretrained(
                model,
                cache_dir=os.path.join(os.getcwd(), "models")
            ).to(self.device)
            print(f"Loaded {model} successfully.")
            self.detection_func = self._detr_detection
            print(f"Set object detection function to {self.detection_func}.")

        elif obj_detection_model == "yolo":
            if YOLO is None:
                raise ImportError(
                    "YOLO inference requires the 'ultralytics' package. "
                    "Install it with 'pip install ultralytics'."
                )

            weights_path = yolo_weights_path or "yolov8n.pt"
            self.yolo_weights_path = weights_path
            self.obj_det_model = YOLO(weights_path)
            self.obj_det_model.to(self.device)
            self.yolo_inference_device = str(self.device)
            self.yolo_fallback_device = "cpu"
            self.detection_func = self._yolo_detection

        else:
            raise ValueError(
                f"Unsupported object detection model: {obj_detection_model}"
            )

        # set to evaluation mode for faster inference
        self.obj_det_model.eval()


    def detect(self, frame, threshold: float = 0.9) -> List[Dict]:
        """Detect objects in the given image frame.

        Args:
            frame: Input image frame (numpy array or PIL Image)
            threshold: Confidence threshold for detections
        Returns:
            List of detected objects with their labels, scores, bounding boxes, and centroids.
        """
        results = self.detection_func(frame, threshold=threshold)
        return results


    def _yolo_detection(self, frame, threshold: float = 0.9) -> List[Dict]:
        if isinstance(frame, Image.Image):
            frame = np.array(frame)

        try:
            results = self.obj_det_model.predict(
                source=frame,
                conf=threshold,
                device=self.yolo_inference_device,
                verbose=False,
            )
        except (NotImplementedError, RuntimeError) as exc:
            message = str(exc)
            if "torchvision::nms" not in message and "CUDA" not in message:
                raise

            if self.yolo_inference_device != self.yolo_fallback_device:
                print(
                    "Warning: YOLO CUDA inference is unavailable in this environment; "
                    "falling back to CPU."
                )
                self.yolo_inference_device = self.yolo_fallback_device

            self.obj_det_model.to(self.yolo_fallback_device)
            results = self.obj_det_model.predict(
                source=frame,
                conf=threshold,
                device=self.yolo_fallback_device,
                verbose=False,
            )

        if not results:
            return []

        prediction = results[0]
        boxes = prediction.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu()
        scores = boxes.conf.detach().cpu()
        class_ids = boxes.cls.detach().cpu().long()
        centroids = (xyxy[:, :2] + xyxy[:, 2:]) / 2.0

        names = prediction.names or getattr(self.obj_det_model, "names", {})

        return [
            {
                "label": names.get(int(class_id), str(int(class_id)))
                if isinstance(names, dict)
                else names[int(class_id)],
                "score": round(float(score), 3),
                "box": torch.round(box, decimals=2).tolist(),
                "centroid": torch.round(centroid, decimals=2).tolist(),
            }
            for box, score, class_id, centroid in zip(xyxy, scores, class_ids, centroids)
        ]


    def _detr_detection(self, frame, threshold: float = 0.9) -> List[Dict]:
        # Preprocess image and perform inference
        inputs = self.processor(
            images=frame, return_tensors="pt").to(self.device)
        # print("Inputs:", inputs)
        outputs = self.obj_det_model(**inputs)
        # print("Outputs:", outputs)

        # Post-process outputs to get detected objects
        # convert outputs (bounding boxes and class logits) to COCO format
        # target_sizes = torch.tensor([frame.shape[::-1]])
        target_sizes = torch.tensor([frame.shape[:2]], device=self.device)
        # print("Target sizes:", target_sizes)
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=threshold
        )[0]
        # print("Results:", results)

        # returns tensors
        boxes = results["boxes"]
        scores = results["scores"]
        labels = results["labels"]

        # vectorized centroid calculation: (xmin, ymin) + (xmax, ymax) / 2
        centroids = (boxes[:, :2] + boxes[:, 2:]) / 2.0

        # convert tensors to lists for easier handling
        boxes_list = torch.round(boxes, decimals=2).tolist()
        scores_list = torch.round(scores, decimals=3).tolist()
        centroids_list = torch.round(centroids, decimals=2).tolist()
        labels_list = labels.tolist()
        id2label = self.obj_det_model.config.id2label

        # Bulk construct the list of dictionaries via standard zip
        detected_objects = [
            {
                "label": id2label[lbl],
                "score": sc,
                "box": bx,
                "centroid": ct
            }
            for lbl, sc, bx, ct in
            zip(labels_list, scores_list, boxes_list, centroids_list)
        ]

        return detected_objects


