#!/usr/bin/env python
"""Example: Basic object detection on a single image."""

import sys
from pathlib import Path
import torch
from PIL import Image
import argparse

# Add project to path for direct script execution
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from alert_system.utils.model_loader import load_detection_model
from alert_system.object_detection.object_detector import ObjectDetector
from alert_system.utils.visualization import draw_detections


def main(image_path: str, threshold: float = 0.9):
    """Run object detection on a single image.

    Args:
        image_path: Path to the image file
        threshold: Confidence threshold for detections
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load models
    print("Loading DETR model...")
    processor, model = load_detection_model(device=device)

    # Create detector
    detector = ObjectDetector(processor, model, device=device)

    # Load image
    image = Image.open(image_path).convert("RGB")
    print(f"Loaded image: {image_path}")
    print(f"Image size: {image.size}\n")

    # Run detection
    print("Running object detection...")
    results = detector.detect(image, threshold=threshold)
    detections = results["objects"]

    # Print results
    print(f"\nDetected {len(detections)} objects:")
    for i, obj in enumerate(detections):
        print(f"  {i+1}. {obj['label']}")
        print(f"     Confidence: {obj['score']}")
        print(f"     Box: {obj['box']}\n")

    # Visualize
    print("Displaying detections...")
    draw_detections(image, detections, show=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run object detection on an image")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--threshold", type=float, default=0.9, help="Detection threshold")

    args = parser.parse_args()
    main(args.image, args.threshold)
