#!/usr/bin/env python
"""Example: Full pipeline with object detection, motion analysis, and LLM commentary."""

import argparse
import os
import torch

from driving_assistant.object_detection.object_detector import ObjectDetector
from driving_assistant.llm.prompt_constructor import PromptConstructor
from driving_assistant.llm.commentary_generator import CommentaryGenerator
from driving_assistant.pipeline.data_pipeline import DataPipeline
from driving_assistant.data_utils.dataset import VideoDataset


def run_pipeline_on_video(
    video_path: str,
    output_dir: str,
    obj_detection_model: str,
    yolo_weights_path: str,
    llm_model_name: str,
    visualize: bool,
) -> None:
    """Run the full vision + LLM pipeline on a single video.
    
    Args:
        video_path: Full path to the video file to process
        output_dir: Directory to save output video
        obj_detection_model: Hugging Face model id for object detection
        yolo_weights_path: Path to YOLO weights file (if using YOLO)
        llm_model_name: Hugging Face model id for commentary generation
        visualize: Whether to show visualizations
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Create pipeline components
    detector = ObjectDetector(obj_detection_model, device=device, yolo_weights_path=yolo_weights_path)
    prompt_constructor = PromptConstructor()
    commentary_generator = CommentaryGenerator(llm_model_name, device=device)

    # Create a minimal dataset with just this video
    # Get the directory and filename
    video_dir = os.path.dirname(video_path)
    
    # Load dataset from the video's directory
    datastream = VideoDataset(video_dir)
    
    # Find the video index
    video_filename = os.path.basename(video_path)
    video_idx = None
    for idx in range(len(datastream)):
        if datastream.get_video_name(idx) == video_filename:
            video_idx = idx
            break
    
    if video_idx is None:
        print(f"Error: Could not find video {video_filename} in dataset")
        return
    
    print(f"Processing video: {video_filename}")
    datastream[video_idx]

    # Create pipeline
    pipeline = DataPipeline(
        datastream=datastream,
        object_detection_model=detector,
        prompt_constructor=prompt_constructor,
        commentary_generator=commentary_generator,
        device=device,
    )
    
    commentary = pipeline.loop(threshold=0.7, visualize=visualize, output_dir=output_dir)

    print("\n" + "="*70)
    print("LLM COMMENTARY")
    print("="*70)
    print(commentary)
    print("="*70)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Run full vision + LLM pipeline on a single video"
    )
    parser.add_argument(
        "--video-path",
        help="Full path to the video file to process",
    )
    parser.add_argument(
        "--output-dir",
        default="output_videos",
        help="Directory to save output video.",
    )
    parser.add_argument(
        "--object-model",
        default="PekingU/rtdetr_r50vd",
        help="Hugging Face model id for object detection.",
    )
    parser.add_argument(
        "--yolo-weights-path",
        default=os.path.join("models", "fine_tuned_yolo_weights", "best.pt"),
        help="Path to YOLO weights file (if using YOLO).",
    )
    parser.add_argument(
        "--llm-model",
        default="google/flan-t5-small",
        help="Hugging Face model id for commentary generation.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show frame visualizations while running.",
    )

    args = parser.parse_args()

    # Make directory for output videos if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    run_pipeline_on_video(
        video_path=args.video_path,
        output_dir=args.output_dir,
        obj_detection_model=args.object_model,
        yolo_weights_path=args.yolo_weights_path,
        llm_model_name=args.llm_model,
        visualize=args.visualize,
    )


if __name__ == "__main__":
    main()
