"""End-to-end data pipeline combining object detection, motion analysis, and LLM commentary."""

from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from driving_assistant.data_utils.dataset import VideoDataset
from driving_assistant.object_detection.object_detector import ObjectDetector
from driving_assistant.llm.prompt_constructor import PromptConstructor
from driving_assistant.llm.commentary_generator import CommentaryGenerator

from driving_assistant.utils.visualization import visualize_frame, Visualizer

# for running the LLM in parallel
from concurrent.futures import ThreadPoolExecutor
import time
import psutil
import os

class DataPipeline:
    """
    A pipeline for processing images and generating commentary using
    a pre-trained object detection model and a small LLM.

    @param datastream: an instance of VideoDataset to read
        video frames.
    @param image_processor: an instance of DetrImageProcessor
        to process images.
    @param object_detection_model: an instance of DetrForObjectDetection
        to detect objects in images.
    @param llm_tokenizer: an instance of AutoTokenizer
        to tokenize prompts for the LLM.
    @param llm_model: an instance of AutoModelForSeq2SeqLM
        to generate commentary for the LLM.
    """

    def __init__(
            self, datastream: VideoDataset,
            object_detection_model: ObjectDetector,
            prompt_constructor: PromptConstructor,
            commentary_generator: CommentaryGenerator,
            device, window_size=10):
        self.obj_detection_model = object_detection_model

        self.prompt_constructor = prompt_constructor
        self.commentary_generator = commentary_generator

        self.window_size = window_size
        self.datastream = datastream
        self.device = device

        # Stores up to window_size frames
        # Each frame is going to be a list of detected objects
        # with their centroids and motion vectors
        self.frames = []
        # this saves the LLM output for each timestep.
        # format of each entry is (timestep, commentary)
        self.llm_commentary = []

        # avg. metrics
        self.avg_detection_time = 0.0
        self.avg_llm_time = 0.0
        self.avg_fps = 0.0
        # tail performance
        self.max_detection_time = 0.0
        self.max_llm_time = 0.0
        self.min_fps = float('inf')

    def _append_frame(self, frame):
        # maintain the sliding window
        if len(self.frames) > self.window_size:
            self.frames.pop(0)

        self.frames.append(frame)

    def reset(self):
        self.frames = []
        self.llm_commentary = []
        self.avg_detection_time = 0.0
        self.avg_llm_time = 0.0
        self.avg_fps = 0.0
        self.max_detection_time = 0.0
        self.max_llm_time = 0.0
        self.min_fps = float('inf')

    def get_metrics(self):
        return {
            "avg_fps": self.avg_fps,
            "avg_detection_time_ms": self.avg_detection_time,
            "avg_llm_time_ms": self.avg_llm_time,
            "max_detection_time_ms": self.max_detection_time,
            "max_llm_time_ms": self.max_llm_time,
            "min_fps": self.min_fps,
        }

    def _compute_motion(self, F_prev, F):
        """
        Compute the motion of each object between two frames.

        It is expected that the frames are lists of detected objects.
        Each detected object is a dictionary with their numerical ID,
        boxes, centroid, and score.

        @param F_prev: list of detected objects in the previous frame.
        @param F: list of detected objects in the current frame.
        """
        if not F:
            return F

        if not F_prev:
            for obj in F:
                obj["velocity"] = (0, 0)
            return F

        # Extract centroids into (K, 2) arrays.
        # N is number of objects in current frame, M is previous frame.
        curr_centroids = np.array(
            [obj["centroid"] for obj in F],
            dtype=np.float32,
        ).reshape(-1, 2)
        prev_centroids = np.array(
            [obj_prev["centroid"] for obj_prev in F_prev],
            dtype=np.float32,
        ).reshape(-1, 2)

        # Compute the distance between object N and M
        # Shape: (N, 1, 2) - (1, M, 2) -> (N, M, 2)
        diff = curr_centroids[:, np.newaxis, :] - \
            prev_centroids[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=2)  # Shape: (N, M)

        # Find index of minimum distance for each object in current frame
        min_dist_indices = np.argmin(dist_matrix, axis=1)
        min_distances = dist_matrix[np.arange(len(F)), min_dist_indices]

        # Assign motion vectors
        threshold = 50
        for i, obj in enumerate(F):
            idx_prev = min_dist_indices[i]
            dist = min_distances[i]

            if dist < threshold:
                # Use the pre-computed 'diff' to get the motion vector
                # diff[i, idx_prev] is (curr - prev)
                obj["velocity"] = tuple(diff[i, idx_prev].tolist())
            else:
                obj["velocity"] = (0, 0)

        return F

    def _timed_llm_generate(self, prompt: str) -> Tuple[str, float]:
        start_time = time.time()
        commentary = self.commentary_generator.generate(prompt)
        end_time = time.time()
        return commentary, end_time - start_time


    def loop(self, visualize=False) -> List[Tuple[int, int, str]]:
        """
        A loop that runs the pipeline
        indefinitely (or until the data stream ends).

        It uses a sliding window for tracking objects.

        @param visualize: if True, visualize the detections and motion vectors
            over time.
        """
        print("Starting data pipeline loop...")
        t = 0
        total_detection_time = 0.0
        total_llm_time = 0.0
        frame_count = 0

        # thread pool for running LLM in parallel with detection
        executor = ThreadPoolExecutor(max_workers=1)
        pending_llm = None

        if visualize:
            h, w = self.datastream.get_height_width()
            visualizer = Visualizer(
                os.path.join(
                    "eval_videos", f"{self.datastream.get_current_video_name()}"),
                height=h, width=w
            )

        metrics = {
            "fps": 0.0,
            "obj_count": 0,
            "avg_det_ms": 0.0,
            "avg_llm_ms": 0.0,
        }

        while True:
            # collect frames
            # push the first frame
            success, frame = self.datastream.step()
            if not success:
                break
            # perform object detection
            # this returns a list of dictionaries with keys
            # "label", "score", "box", and "centroid"
            detection_start = time.time()
            detected_obj = self.obj_detection_model.detect(frame)
            total_detection_time += time.time() - detection_start

            self._append_frame(detected_obj)
            frame_count += 1

            for i in range(self.window_size):
                success, frame = self.datastream.step()
                if not success:
                    break

                detection_start = time.time()
                detected_obj = self.obj_detection_model.detect(frame)
                total_detection_time += time.time() - detection_start

                detected_obj = self._compute_motion(self.frames[-1], detected_obj)
                self._append_frame(detected_obj)
                frame_count += 1
                total_time = total_detection_time

                if visualize:
                    metrics = {
                        "fps": frame_count / total_time if total_time > 0 else 0,
                        "obj_count": len(detected_obj),
                        "avg_det_ms": (total_detection_time / frame_count) * 1000,
                        "avg_llm_ms": (total_llm_time / max(t, 1)) * 1000,
                    }
                    self.avg_fps += metrics["fps"]
                    self.avg_detection_time += metrics["avg_det_ms"]
                    self.avg_llm_time += metrics["avg_llm_ms"]
                    self.max_detection_time = max(self.max_detection_time, metrics["avg_det_ms"])
                    self.max_llm_time = max(self.max_llm_time, metrics["avg_llm_ms"])
                    self.min_fps = min(self.min_fps, metrics["fps"])

                    visualizer.update(
                        frame, detected_obj,
                        metrics=metrics,
                        commentary=self.llm_commentary[-1][2] if len(self.llm_commentary) > 0 else None
                    )

            # initialize prompt generation
            prompt = self.prompt_constructor.generate_prompt(
                self.frames,
                t=t - len(self.frames)
            )
            if pending_llm is not None:
                # commentary is a dictionary with keys "warning" (bool) and "message" (str)
                commentary, llm_elapsed = pending_llm.result()
                total_llm_time += llm_elapsed
                
                if commentary['warning']:
                    timestamp = self.datastream.get_current_time()
                    self.llm_commentary.append((t, timestamp, commentary["message"]))
            else:
                # start inference
                pending_llm = executor.submit(self._timed_llm_generate, prompt)

            t += 1

        #Printing Metrics
            # print all metrics
            # if frame_count > 0:
            #     total_time = total_detection_time + total_llm_time
            #     fps = frame_count / total_time if total_time > 0 else 0
            #     avg_objects = sum(len(f) for f in self.frames) / max(len(self.frames), 1)
            #     avg_confidence = sum(all_scores) / len(all_scores) if all_scores else 0
            #     ram_used = psutil.Process().memory_info().rss / 1024 ** 2

            #     print(f"\nPipeline Metrics")
            #     print(f"Total frames processed:        {frame_count}")
            #     print(f"Total pipeline time:           {total_time:.2f}s")
            #     print(f"Pipeline FPS:                  {fps:.2f}")
            #     print(f"Total detection time:          {total_detection_time:.2f}s")
            #     print(f"Avg detection time per frame:  {total_detection_time / frame_count:.3f}s")
            #     print(f"Total LLM time:                {total_llm_time:.2f}s")
            #     print(f"Avg LLM time per window:       {total_llm_time / max(t, 1):.3f}s")
            #     print(f"Avg objects detected per frame:{avg_objects:.1f}")
            #     print(f"Avg detection confidence:      {avg_confidence:.2%}")
            #     print(f"RAM used:                      {ram_used:.1f} MB")

            #     if torch.cuda.is_available():
            #         gpu_used = torch.cuda.memory_allocated() / 1024 ** 2
            #         print(f"GPU memory used:               {gpu_used:.1f} MB")

        if visualize:
            visualizer.release()
            # cv2.destroyAllWindows()

        self.avg_fps /= t
        self.avg_detection_time /= t
        self.avg_llm_time /= t

        return self.llm_commentary
