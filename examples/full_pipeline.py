#!/usr/bin/env python
"""Example: Full pipeline with object detection, motion analysis, and LLM commentary."""

import random
import torch
from PIL import Image
import argparse
import os

from driving_assistant.object_detection.object_detector import ObjectDetector
from driving_assistant.llm.prompt_constructor import PromptConstructor
from driving_assistant.llm.commentary_generator import CommentaryGenerator
from driving_assistant.pipeline.data_pipeline import DataPipeline
from driving_assistant.data_utils.dataset import VideoDataset

def main(threshold: float = 0.9):
    """Run the full vision + LLM pipeline on a video.

    Args:
        threshold: Confidence threshold for detections
    """

    keywords = ["pedestrian", "vehicle", "bicycle", "motorcycle", "traffic light", "stop sign"]

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Create pipeline components
    detector = ObjectDetector("PekingU/rtdetr_r50vd", device=device)
    prompt_constructor = PromptConstructor(keywords)
    commentary_generator = CommentaryGenerator("google/flan-t5-small", device=device)

    # load dataset
    path = os.path.join(os.getcwd(), "data", "reddit_dashcam_videos")
    datastream = VideoDataset(path)
    # open a random video
    idx = random.randint(0, len(datastream)-1)
    print(f"Selected video {idx} for processing: {datastream.get_video_name(idx)}")
    datastream[idx]

    # Create pipeline
    pipeline = DataPipeline(
        datastream=datastream,
        object_detection_model=detector,
        prompt_constructor=prompt_constructor,
        commentary_generator=commentary_generator,
        device=device
    )
    commentary = pipeline.loop(visualize=True)

    print("\n" + "="*70)
    print("LLM COMMENTARY")
    print("="*70)
    print(commentary)
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run full vision + LLM pipeline on a video"
    )
    parser.add_argument("--threshold", type=float, default=0.9, help="Detection threshold")

    args = parser.parse_args()
    main(args.threshold)
