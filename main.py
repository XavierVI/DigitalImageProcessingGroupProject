"""Run the full driving assistant pipeline over all videos and report metrics."""

import argparse
import json
import os
import re
import time
from typing import Dict, List

import torch

from driving_assistant.data_utils.dataset import VideoDataset
from driving_assistant.llm.commentary_generator import CommentaryGenerator
from driving_assistant.llm.prompt_constructor import PromptConstructor
from driving_assistant.object_detection.object_detector import ObjectDetector
from driving_assistant.pipeline.data_pipeline import DataPipeline


def calculate_metrics(manual_labels: Dict[str, List[str]], model_outputs: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
    """
    manual_labels: dict { "vid_1": ["red_light", "night"] }
    model_outputs: dict { "vid_1": ["red_light", "intersection"] }

    The outputs can just be the words split up by spaces.
    ""
    results = {}

    for vid, gt_tags in manual_labels.items():
        if vid not in model_outputs:
            continue

        pred_tags = set(model_outputs[vid])
        gt_tags = set(gt_tags)

        # Intersection over Union (Jaccard)
        intersection = gt_tags.intersection(pred_tags)
        union = gt_tags.union(pred_tags)
        jaccard = len(intersection) / len(union) if union else 0

        # Precision (How many predicted tags were right?)
        precision = len(intersection) / len(pred_tags) if pred_tags else 0

        # Recall (How many ground truth tags did we find?)
        recall = len(intersection) / len(gt_tags) if gt_tags else 0

        results[vid] = {
            "jaccard": round(jaccard, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3)
        }

    return results


def _normalize_tag(tag: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(tag).lower()).strip()
    normalized = re.sub(r"\s+", " ", normalized)

    replacements = {
        "braking": "breaking",
        "cutting off": "cut off",
        "changing lane": "changing lanes",
        "change lane": "change lanes",
        "slowing down": "slow down",
    }

    return replacements.get(normalized, normalized)


def build_label_vocabulary(manual_labels: Dict[str, List[str]]) -> set[str]:
    vocabulary: set[str] = set()
    for tags in manual_labels.values():
        for tag in tags:
            normalized = _normalize_tag(tag)
            if normalized:
                vocabulary.add(normalized)
    return vocabulary


def _normalize_commentary_text(commentary: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", commentary.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()

    replacements = {
        "braking": "breaking",
        "cutting off": "cut off",
        "changing lane": "changing lanes",
        "change lane": "change lanes",
        "slowing down": "slow down",
    }

    for source, target in replacements.items():
        normalized = normalized.replace(source, target)

    return normalized


def _normalize_token(token: str, vocabulary: set[str]) -> str:
    if token in vocabulary:
        return token
    if token.endswith("ies") and len(token) > 3:
        candidate = token[:-3] + "y"
        if candidate in vocabulary:
            return candidate
    if token.endswith("es") and len(token) > 4:
        candidate = token[:-2]
        if candidate in vocabulary:
            return candidate
    if token.endswith("s") and len(token) > 3:
        candidate = token[:-1]
        if candidate in vocabulary:
            return candidate
    return token


def extract_tags_from_commentary(llm_output: List[tuple[int, str]], vocabulary: set[str]) -> List[str]:
    """Extract normalized alert tags from generated commentary text."""
    tags: List[str] = []
    seen: set[str] = set()
    text = _normalize_commentary_text(" ".join(commentary for _, commentary in llm_output))

    phrase_tags = [tag for tag in vocabulary if " " in tag]
    phrase_tags.sort(key=lambda tag: (-len(tag.split()), -len(tag)))

    for phrase in phrase_tags:
        if re.search(rf"(?<!\\w){re.escape(phrase)}(?!\\w)", text) and phrase not in seen:
            tags.append(phrase)
            seen.add(phrase)

    token_tags = {
        _normalize_token(token, vocabulary)
        for token in re.findall(r"[a-z0-9_'-]+", text)
    }

    for token in sorted(token_tags):
        if token in vocabulary and token not in seen:
            tags.append(token)
            seen.add(token)

    return tags


def load_labels(labels_path: str) -> Dict[str, List[str]]:
    """Load manual labels from JSON if the file exists."""
    if not os.path.exists(labels_path):
        return {}

    with open(labels_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    normalized = {}
    for video_name, tags in data.items():
        normalized[video_name] = [str(tag).strip().lower() for tag in tags]
    return normalized


def summarize_metrics(per_video_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Compute dataset-level averages for jaccard, precision, and recall."""
    if not per_video_metrics:
        return {"jaccard": 0.0, "precision": 0.0, "recall": 0.0}

    count = len(per_video_metrics)
    return {
        "jaccard": round(sum(m["jaccard"] for m in per_video_metrics.values()) / count, 3),
        "precision": round(sum(m["precision"] for m in per_video_metrics.values()) / count, 3),
        "recall": round(sum(m["recall"] for m in per_video_metrics.values()) / count, 3),
    }


def summarize_runtime_metrics(per_video_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    if not per_video_metrics:
        return {
            "fps": 0.0,
            "wall_time_s": 0.0,
            "avg_detection_ms": 0.0,
            "avg_llm_ms": 0.0,
        }

    count = len(per_video_metrics)
    return {
        "fps": round(sum(m.get("fps", 0.0) for m in per_video_metrics.values()) / count, 3),
        "wall_time_s": round(sum(m.get("wall_time_s", 0.0) for m in per_video_metrics.values()) / count, 3),
        "avg_detection_ms": round(sum(m.get("avg_detection_ms", 0.0) for m in per_video_metrics.values()) / count, 3),
        "avg_llm_ms": round(sum(m.get("avg_llm_ms", 0.0) for m in per_video_metrics.values()) / count, 3),
    }


def run_pipeline_over_dataset(
    video_dir: str,
    labels_path: str,
    obj_detection_model: str,
    llm_model_name: str,
    visualize: bool,
    max_videos: int | None = None,
) -> None:
    """Initialize pipeline components and evaluate all videos in the dataset."""
    keywords = [
        # alert keywords
        "caution", "warning", "attention", "danger", "watch out",

        # object keywords
        "pedestrian", "vehicle", "bicycle", "motorcycle",
        "traffic light", "stop sign", "red light", "green light", "yellow light",

        # action keywords
        "slow down", "speed up", "turn left",
        "turn right", "stop", "go",

        # directions
        "ahead", "left", "right", "behind",
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    detector = ObjectDetector(obj_detection_model, device=device)
    prompt_constructor = PromptConstructor(keywords)
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
    label_vocabulary = build_label_vocabulary(manual_labels)
    model_outputs: Dict[str, List[str]] = {}
    runtime_metrics: Dict[str, Dict[str, float]] = {}

    total_videos = len(dataset) if max_videos is None else min(len(dataset), max_videos)
    print(f"Found {len(dataset)} videos. Running {total_videos} video(s)...")
    for idx in range(total_videos):
        video_name = dataset.get_video_name(idx)
        is_opened, _ = dataset[idx]
        if not is_opened:
            print(f"Skipping {video_name}: failed to open video.")
            continue

        video_start = time.perf_counter()
        pipeline.reset()
        llm_output = pipeline.loop(visualize=visualize)
        model_outputs[video_name] = extract_tags_from_commentary(llm_output, label_vocabulary)
        runtime_metrics[video_name] = {
            **pipeline.last_runtime_metrics,
            "wall_time_s": round(time.perf_counter() - video_start, 3),
        }
        print(f"{llm_output}")
        print(f"Processed {idx + 1}/{len(dataset)}: {video_name}")

    print("\nPipeline run complete.")

    runtime_summary = summarize_runtime_metrics(runtime_metrics)
    print("\nRuntime benchmark:")
    for video_name, metrics in runtime_metrics.items():
        print(f"- {video_name}: {metrics}")
    print("\nBenchmark averages:")
    print(runtime_summary)

    if not manual_labels:
        print(f"No labels found at {labels_path}. Skipping metric computation.")
        return

    per_video_metrics = calculate_metrics(manual_labels, model_outputs)
    final_metrics = summarize_metrics(per_video_metrics)

    print("\nPer-video metrics:")
    for video_name, metrics in per_video_metrics.items():
        print(f"- {video_name}: {metrics}")

    print("\nFinal metrics:")
    print(final_metrics)


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
        "--max-videos",
        type=int,
        default=None,
        help="Limit how many videos to process. Omit to process all videos.",
    )
    parser.add_argument(
        "--max-video",
        type=int,
        default=None,
        help="Alias for --max-videos.",
    )

    args = parser.parse_args()
    run_pipeline_over_dataset(
        video_dir=args.video_dir,
        labels_path=args.labels_path,
        obj_detection_model=args.object_model,
        llm_model_name=args.llm_model,
        visualize=args.visualize,
        max_videos=args.max_videos if args.max_videos is not None else args.max_video,
    )


if __name__ == "__main__":
    main()