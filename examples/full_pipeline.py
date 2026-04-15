#!/usr/bin/env python
"""Example: Full pipeline with object detection, motion analysis, and LLM commentary."""

import sys
from pathlib import Path
import torch
from PIL import Image
import argparse

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from alert_system.utils.model_loader import load_detection_model, load_llm_model
from alert_system.object_detection.object_detector import ObjectDetector
from alert_system.llm.prompt_constructor import PromptConstructor
from alert_system.llm.commentary_generator import CommentaryGenerator
from alert_system.pipeline.data_pipeline import DataPipeline


def main(image_path: str, threshold: float = 0.9):
    """Run the full vision + LLM pipeline on an image.

    Args:
        image_path: Path to the image file
        threshold: Confidence threshold for detections
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load models
    print("Loading DETR detection model...")
    detection_processor, detection_model = load_detection_model(device=device)

    print("Loading LLM model...")
    llm_tokenizer, llm_model = load_llm_model(device=device)

    # Create pipeline components
    detector = ObjectDetector(detection_processor, detection_model, device=device)
    prompt_constructor = PromptConstructor(context="driver assistance system")
    commentary_generator = CommentaryGenerator(llm_tokenizer, llm_model, device=device)

    # Create pipeline
    pipeline = DataPipeline(
        object_detector=detector,
        prompt_constructor=prompt_constructor,
        commentary_generator=commentary_generator,
        detection_threshold=threshold
    )

    # Load image
    image = Image.open(image_path).convert("RGB")
    print(f"\nLoaded image: {image_path}")
    print(f"Image size: {image.size}\n")

    # Process frame
    print("Processing frame through pipeline...")
    result = pipeline.process_frame(image)

    # Print results
    print("\n" + "="*70)
    print("DETECTION RESULTS")
    print("="*70)
    detections = result["detections"]
    print(f"Found {len(detections)} objects:")
    for i, obj in enumerate(detections):
        print(f"  {i+1}. {obj['label']} (confidence: {obj['score']})")

    print("\n" + "="*70)
    print("LLM PROMPT")
    print("="*70)
    print(result["prompt"])

    print("\n" + "="*70)
    print("LLM COMMENTARY")
    print("="*70)
    print(result["commentary"])
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run full vision + LLM pipeline on an image"
    )
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--threshold", type=float, default=0.9, help="Detection threshold")

    args = parser.parse_args()
    main(args.image, args.threshold)
