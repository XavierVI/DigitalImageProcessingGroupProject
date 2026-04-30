#!/usr/bin/env python
"""Example: Combined vehicle and traffic sign detection."""

import sys
from pathlib import Path
import torch
from PIL import Image
import argparse
import os

from driving_assistant.object_detection.combined_detector import CombinedDetector
from driving_assistant.utils.visualization import draw_detections


def main(
    image_path: str,
    sign_weights_path: str,
    vehicle_threshold: float = 0.9,
    sign_threshold: float = 0.5
):
    """Run combined vehicle and traffic sign detection on a single image.

    Args:
        image_path: Path to image file
        sign_weights_path: Path to YOLOv8 traffic signs weights (.pt)
        vehicle_threshold: Confidence threshold for vehicle detection
        sign_threshold: Confidence threshold for sign detection
    """
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Create combined detector
    # - Vehicle detector: Uses DETR model (general object detection)
    # - Sign detector: Uses YOLOv8 fine-tuned on traffic signs
    print("Loading vehicle detector (DETR)...")
    print(f"Loading sign detector (YOLOv8) from {sign_weights_path}...")
    
    detector = CombinedDetector(
        vehicle_model="PekingU/rtdetr_r50vd",  # DETR for vehicles
        sign_model="yolo",
        sign_yolo_weights=sign_weights_path,  # Your fine-tuned YOLOv8
        device=device
    )

    # Load image
    image = Image.open(image_path).convert("RGB")
    print(f"Loaded image: {image_path}")
    print(f"Image size: {image.size}\n")

    # Run combined detection
    print("Running combined detection...")
    results = detector.detect(
        image,
        vehicle_threshold=vehicle_threshold,
        sign_threshold=sign_threshold
    )

    if len(results) == 0:
        print("No objects detected above the threshold.")
        return

    # Print results organized by type
    print(f"\nDetected {len(results)} objects total:\n")
    
    vehicles = [r for r in results if r.get('detection_type') == 'vehicle']
    signs = [r for r in results if r.get('detection_type') == 'traffic_sign']
    
    if vehicles:
        print(f"Vehicles ({len(vehicles)}):")
        for i, obj in enumerate(vehicles):
            print(f"  {i+1}. {obj['label']}")
            print(f"     Confidence: {obj['score']:.2%}")
            print(f"     Box: {obj['box']}")
            if obj.get('velocity'):
                vx, vy = obj['velocity']
                speed = (vx**2 + vy**2)**0.5
                print(f"     Speed: {speed:.1f} px/frame\n")
    
    if signs:
        print(f"\nTraffic Signs ({len(signs)}):")
        for i, obj in enumerate(signs):
            print(f"  {i+1}. {obj['label']}")
            print(f"     Confidence: {obj['score']:.2%}")
            print(f"     Box: {obj['box']}\n")

    # Visualize
    print("Displaying detections...")
    print("  Blue boxes = Vehicles")
    print("  Orange boxes = Traffic Signs")
    draw_detections(image, results, show=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run combined vehicle and traffic sign detection on an image"
    )
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument(
        "--sign-weights",
        required=True,
        help="Path to YOLOv8 traffic signs weights (.pt)"
    )
    parser.add_argument(
        "--vehicle-threshold",
        type=float,
        default=0.9,
        help="Confidence threshold for vehicle detection (default: 0.9)"
    )
    parser.add_argument(
        "--sign-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for traffic sign detection (default: 0.5)"
    )

    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)
    
    if not os.path.exists(args.sign_weights):
        print(f"Error: Sign weights file not found: {args.sign_weights}")
        sys.exit(1)

    main(
        args.image,
        args.sign_weights,
        args.vehicle_threshold,
        args.sign_threshold
    )
