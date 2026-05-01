"""Run the full driving assistant pipeline over all videos and report metrics."""

import argparse
import json
import os
import re
from typing import Dict, List

import torch

from driving_assistant.data_utils.dataset import VideoDataset
from driving_assistant.llm.commentary_generator import CommentaryGenerator
from driving_assistant.llm.prompt_constructor import PromptConstructor
from driving_assistant.object_detection.object_detector import ObjectDetector
from driving_assistant.pipeline.data_pipeline import DataPipeline


RISK_KEYWORDS = {
    "cut-in": ["cut", "lane change", "merg"],
    "tailgating": ["tailgat", "following", "too close"],
    "pedestrian": ["pedestrian", "person", "crossing"],
    "braking": ["brak", "slow", "stopping"],
    "head-on": ["head-on", "oncoming", "collision"],
    "cross-traffic": ["cross", "intersection", "traffic"],
    "animal": ["animal", "deer", "dog"],
    "speeding": ["speed", "fast", "rapid"],
}


def message_matches_risk(risk_label: str, message: str) -> bool:
    message = message.lower()
    for risk_type, keywords in RISK_KEYWORDS.items():
        if risk_type in risk_label.lower():
            return any(kw in message for kw in keywords)
    return False


def calculate_metrics(
    manual_labels: Dict[List[int, int], str],
    model_outputs: Dict[str, List[int, int, str]]) -> List[int, int, int, int]:
    """
    computes model performance metrics.

    Returns a dictionary with keys
    true positives (TP), true negatives (TN), false positives (FP), false negatives (FN).
    """
    results = {
        "TP": 0,
        "TN": 0,
        "FP": 0,
        "FN": 0,
    }

    for vid, _ in manual_labels.items():
        expected_timestamp = manual_labels[vid]['timestamps']
        expected_keywords = manual_labels[vid]['risk']
        if vid not in model_outputs:
            continue

        if len(expected_timestamp) == 0: # nothing occurred
            if len(model_outputs[vid]) == 0: # nothing predicted
                results["TN"] += 1
            else: # something predicted
                results["FP"] += len(model_outputs[vid])
            continue

        # interval for where we expect a warning
        t0, t1 = expected_timestamp

        if len(model_outputs[vid]) == 0: # nothing predicted
            results["FN"] += 1
            continue
        
        for _, timestamp, message in model_outputs[vid]:
            if t0 <= timestamp and timestamp <= t1:
                # check if the message contains expected keywords for the risk type
                if message_matches_risk(expected_keywords, message):
                    results["TP"] += 1
                else: # message doesn't match expected risk keywords, count as false positive
                    results["FP"] += 1
            else:
                results["FP"] += 1

    # normalize results to get precision, recall, jaccard
    tp = results["TP"]
    tn = results["TN"]
    fp = results["FP"]
    fn = results["FN"]

    results["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    results["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    results["jaccard"] = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return results


def save_perf_metrics(perf_metrics: Dict[str, Dict[str, float]], output_path: str) -> None:
    """Save performance metrics to a JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(perf_metrics, f, indent=4)


def load_labels(labels_path: str):
    """Load manual labels from JSON if the file exists."""
    if not os.path.exists(labels_path):
        return {}

    with open(labels_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data


def run_pipeline_over_dataset(
    video_dir: str,
    output_dir: str,
    labels_path: str,
    obj_detection_model: str,
    yolo_weights_path: str,
    llm_model_name: str,
    visualize: bool,
    max_videos: int = None,
    window_size: int = 10,
) -> None:
    """Initialize pipeline components and evaluate all videos in the dataset."""
    keywords = [
        # alert keywords
        # "caution", "warning", "attention", "danger", "watch out",

        # object keywords
        "pedestrian", "vehicle", "bicycle", "motorcycle",
        "traffic light", "stop sign", "red light", "green light", "yellow light",

        # action keywords
        # "slow down", "speed up", "turn left",
        # "turn right", "stop", "go",

        # directions
        # "ahead", "left", "right", "behind",
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    detector = ObjectDetector(obj_detection_model, device=device, yolo_weights_path=yolo_weights_path)
    prompt_constructor = PromptConstructor()#keywords=keywords)
    commentary_generator = CommentaryGenerator(llm_model_name, device=device)

    dataset = VideoDataset(root_dir=video_dir)
    pipeline = DataPipeline(
        datastream=dataset,
        object_detection_model=detector,
        prompt_constructor=prompt_constructor,
        commentary_generator=commentary_generator,
        device=device,
    )

    manual_labels = load_labels(labels_path)
    print(f"Loaded manual labels for {len(manual_labels)} videos from {labels_path}.")
    model_outputs: Dict[str, List[str]] = {}
    perf_metrics = {}

    print(f"Found {len(dataset)} videos. Running full pipeline...")
    
    if max_videos is not None:
        print(f"Limiting to max {max_videos} videos for testing.")
    else:
        max_videos = len(dataset)
    
    for idx in range(max_videos):
        video_name = dataset.get_video_name(idx)
        is_opened, _ = dataset[idx]
        if not is_opened:
            print(f"Skipping {video_name}: failed to open video.")
            continue

        pipeline.reset()
        llm_output = pipeline.loop(visualize=visualize, output_dir=output_dir)
        model_outputs[video_name] = llm_output
        perf_metrics[video_name] = pipeline.get_metrics()
        print(f"Processed {idx + 1}/{max_videos}: {video_name}")

    print("\nPipeline run complete.")

    if not manual_labels:
        print(f"No labels found at {labels_path}. Skipping metric computation.")
        return

    results = calculate_metrics(manual_labels, model_outputs)
    perf_metrics["accuracy_metrics"] = results
    save_perf_metrics(perf_metrics, os.path.join(output_dir, "performance_metrics.json"))
    print("\nEvaluation Metrics:")
    print(f"True Positives: {results['TP']}")
    print(f"True Negatives: {results['TN']}")
    print(f"False Positives: {results['FP']}")
    print(f"False Negatives: {results['FN']}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"Jaccard Index: {results['jaccard']:.4f}")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Run the full driving assistant pipeline on all videos and compute final metrics."
    )
    parser.add_argument(
        "--video-dir",
        default=os.path.join("data", "reddit_dashcam_videos"),
        help="Directory containing input videos.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("eval_videos"),
        help="Directory to save output videos and metrics.",
    )
    parser.add_argument(
        "--max-videos",
        default=None,
        type=int,
        help="Maximum number of videos to process.",
    )
    parser.add_argument(
        "--labels-path",
        default=os.path.join("data", "reddit_dashcam_videos", "labels.json"),
        help="Path to manual labels JSON generated by annotator.py.",
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
    parser.add_argument(
        "--window-size",
        default=10,
        type=int,
        help="Time window size in seconds for matching predictions to labels when computing metrics.",
    )

    args = parser.parse_args()

    # make directory for output videos if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    run_pipeline_over_dataset(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        labels_path=args.labels_path,
        obj_detection_model=args.object_model,
        yolo_weights_path=args.yolo_weights_path,
        llm_model_name=args.llm_model,
        visualize=args.visualize,
        max_videos=args.max_videos,
        window_size=args.window_size,
    )


if __name__ == "__main__":
    main()