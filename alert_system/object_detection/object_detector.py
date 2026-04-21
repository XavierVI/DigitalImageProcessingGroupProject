"""Object detection using DETR model."""

import torch
from PIL import Image
from typing import List, Dict, Optional


class ObjectDetector:
    """Performs object detection on images using pre-trained DETR model.

    Args:
        image_processor: Processor for preparing images for the model
        model: Pre-trained DETR model
        device: Device to run inference on (cuda or cpu)
    """

    def __init__(self, image_processor, model, device=None):
        """Initialize the object detector.

        Args:
            image_processor: Transformer image processor from Hugging Face
            model: DETR model from Hugging Face
            device: torch.device to use for inference
        """
        self.image_processor = image_processor
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def detect(self, image: Image.Image, threshold: float = 0.9) -> Dict:
        """Detect objects in an image.

        Args:
            image (PIL.Image): Input image in RGB format
            threshold (float): Confidence threshold for detections (0-1)

        Returns:
            dict: Contains:
                - 'objects': List of detected objects with labels, scores, and boxes
                - 'raw_results': Raw model outputs for advanced usage
        """
        # Prepare input
        inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process outputs
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.image_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=threshold
        )[0]

        # Extract detected objects
        detected_objects = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box_list = [round(coord, 2) for coord in box.tolist()]
            centroid = self._compute_centroid(box_list)
            detected_objects.append({
                "label": self.model.config.id2label[label.item()],
                "score": round(score.item(), 3),
                "box": box_list,  # [xmin, ymin, xmax, ymax]
                "centroid": centroid,
            })

        return {
            "objects": detected_objects,
            "raw_results": results
        }

    def detect_batch(self, image: Image.Image, threshold: float = 0.9) -> Dict:
        """Detect objects in multiple images.

        Args:
            images: List of PIL Images
            threshold: Confidence threshold for detections

        Returns:
            list: List of detection results (one dict per image)
        """
        result = self.detect(image, threshold)
        objects = result["objects"]

        label_counts: Dict[str, int] = {}
        total_confidence = 0.0

        for obj in objects:
            lbl = obj["label"]
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
            total_confidence += obj["score"]

        avg_confidence = (
            round(total_confidence / len(objects), 3) if objects else 0.0
        )

        result["stats"] = {
            "total objects": len(objects),
            "unique labels": list(label_counts.keys()),
            "avg confidence": avg_confidence,
            "label counts": label_counts,
        }
        return result

    def get_objects_by_label(
            self,
            image: Image.Image,
            label: str,
            threshold: float = 0.9,
    ) -> List[Dict]:
        """Return only detections that match a specific label.
        Args:
            image (PIL.Image): Input image.
            label (str): Class name to filter by (case-insensitive).
            threshold (float): Confidence threshold.

        Returns:
            list[dict]: Filtered detection list.
        """
        result = self.detect(image, threshold)
        return [
            obj for obj in result["objects"]
            if obj["label"].lower() == label.lower()
        ]

    def _compute_centroid(self, box: List[float]) -> List[float]:
        """Compute the centroid of a bounding box.
        Args:
            box (list): [xmin, ymin, xmax, ymax] in pixel coordinates.

        Returns:
            list: [cx, cy] rounded to 2 decimal places.
        """
        xmin, ymin, xmax, ymax = box
        cx = round((xmin + xmax) / 2, 2)
        cy = round((ymin + ymax) / 2, 2)
        return [cx, cy]
