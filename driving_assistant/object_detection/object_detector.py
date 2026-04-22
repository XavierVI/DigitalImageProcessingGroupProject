"""Object detection using DETR model."""

import torch
from PIL import Image
from typing import List, Dict, Optional

from transformers import (
    DetrImageProcessor,
    DetrForObjectDetection,
    RTDetrImageProcessor,
    RTDetrForObjectDetection,
)

import os

class ObjectDetector:
    """Performs object detection on images using pre-trained DETR model.

    Args:
        processor: Processor for preparing images for the model
        model: Pre-trained DETR model
        device: Device to run inference on (cuda or cpu)
    """

    def __init__(self, obj_detection_model, device=None):
        """Initialize the object detector.

        Args:
            obj_detection_model: Pre-trained model to load.
                (e.g., "facebook/detr-resnet-50" or "PekingU/rtdetr_r50vd" or "yolo")
            device: torch.device to use for inference
        """
        self.device = device
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
            ).to(device)
            self.detection_func = self._detr_detection

        elif obj_detection_model == "PekingU/rtdetr_r50vd":
            self.processor = RTDetrImageProcessor.from_pretrained(
                model,
                cache_dir=os.path.join(os.getcwd(), "models")
            )
            self.obj_det_model = RTDetrForObjectDetection.from_pretrained(
                model,
                cache_dir=os.path.join(os.getcwd(), "models")
            ).to(device)
            print(f"Loaded {model} successfully.")
            self.detection_func = self._detr_detection
            print(f"Set object detection function to {self.detection_func}.")

        elif obj_detection_model == "yolo":
            # Initialize YOLO model (example, replace with actual YOLO initialization)
            pass
            self.detection_func = self._yolo_detection

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
        print(f"Detection function {self.detection_func.__name__} returned {len(results)} objects.")
        return results


    def _yolo_detection(self, frame, threshold: float = 0.9) -> List[Dict]:
        # Placeholder for YOLO detection logic
        # Implement YOLO detection and return results in the same format as DETR
        pass


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
        print(f"Detected {len(detected_objects)} objects: {detected_objects}")

        return detected_objects


