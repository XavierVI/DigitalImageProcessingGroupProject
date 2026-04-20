"""End-to-end data pipeline combining object detection, motion analysis, and LLM commentary."""

from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from alert_system.data_stream.dataset import VideoDataset
from alert_system.object_detection.object_detector import ObjectDetector
from alert_system.pipeline.motion_analyzer import MotionAnalyzer
from alert_system.llm.prompt_constructor import PromptConstructor
from alert_system.llm.commentary_generator import CommentaryGenerator


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
            image_processor, object_detection_model,
            llm_tokenizer, llm_model, device,
            window_size=10):
        self.image_processor = image_processor
        self.object_detection_model = object_detection_model

        self.llm_tokenizer = llm_tokenizer
        self.llm_model = llm_model

        self.window_size = window_size
        self.datastream = datastream
        self.device = device

        # Stores up to window_size frames
        # Each frame is going to be a list of detected objects
        # with their centroids and motion vectors
        self.frames = []

    def _append_frame(self, frame):
        # maintain the sliding window
        if len(self.frames) > self.window_size:
            self.frames.pop(0)

        self.frames.append(frame)

    def _visualize_frame(self, frame, detected_obj):
        # Draw into a copy so we do not mutate the original frame.
        annotated = frame.copy()

        for obj in detected_obj:
            bx = [int(i) for i in obj["box"]]
            cv2.rectangle(
                annotated,
                (bx[0], bx[1]),
                (bx[2], bx[3]),
                (255, 0, 0),
                2,
            )

            c = tuple(map(int, obj["centroid"]))
            m = obj.get("velocity", (0, 0))
            end_point = (int(c[0] + m[0]), int(c[1] + m[1]))
            if end_point != c:
                cv2.arrowedLine(
                    annotated,
                    c,
                    end_point,
                    (0, 255, 0),
                    2,
                    tipLength=0.3,
                )

            cv2.putText(
                annotated,
                str(obj["label"]),
                (bx[0], max(20, bx[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

        # Fast real-time rendering for video playback.
        cv2.imshow("Alert System", annotated)
        cv2.waitKey(1)

    def _obj_detection(self, frame):
        # Preprocess image and perform inference
        inputs = self.image_processor(
            images=frame, return_tensors="pt").to(self.device)
        # print("Inputs:", inputs)
        outputs = self.object_detection_model(**inputs)
        # print("Outputs:", outputs)

        # Post-process outputs to get detected objects
        # convert outputs (bounding boxes and class logits) to COCO format
        # target_sizes = torch.tensor([frame.shape[::-1]])
        target_sizes = torch.tensor([frame.shape[:2]], device=self.device)
        # print("Target sizes:", target_sizes)
        results = self.image_processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=0.9
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
        id2label = self.object_detection_model.config.id2label

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

        return detected_objects

    def _compute_motion(self, F_prev, F):
        """
        Compute the motion of each object between two frames.

        It is expected that the frames are lists of detected objects.
        Each detected object is a dictionary with their numerical ID,
        boxes, centroid, and score.

        @param F_prev: list of detected objects in the previous frame.
        @param F: list of detected objects in the current frame.
        """
        # Extract centroids into (K, 2) arrays
        # N is number of objects in current frame, M is previous frame
        curr_centroids = np.array([obj["centroid"] for obj in F])
        prev_centroids = np.array([obj_prev["centroid"]
                                  for obj_prev in F_prev])

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

    def build_prompt(self):
        template = "Generate commentary for the following traffic scene:\n\n"
        for i, frame in enumerate(self.frames):
            template += f"Frame {i+1}:\n"
            for obj in frame:
                label = obj['label']
                score = obj['score']
                velocity = obj.get('velocity', (0, 0))
                template += f"- Detected a {label} with confidence {score:.2f} and velocity {velocity}\n"
            template += "\n"
        template += "Provide a concise summary of the traffic scene, including any notable events or interactions between objects."
        template += "Warn the user of any potential hazards or interesting occurrences in the scene."
        return template

    def generate_commentary(self):
        """
        Generate commentary using time series data.
        """
        # build prompt
        prompt = self.build_prompt()

        # print("Prompt for LLM:", prompt)
        # tokenize and generate commentary
        inputs = self.llm_tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.llm_model.generate(**inputs, max_length=200)
        commentary = self.llm_tokenizer.decode(
            outputs[0], skip_special_tokens=True)
        return commentary

    def loop(self, visualize=False):
        """
        A loop that runs the pipeline
        indefinitely (or until the data stream ends).

        It uses a sliding window for tracking objects.

        @param visualize: if True, visualize the detections and motion vectors
            over time.
        """
        print("Starting data pipeline loop...")
        while True:
            # collect frames
            # push the first frame
            success, frame = self.datastream.step()
            if not success:
                break
            # perform object detection
            # this returns a list of dictionaries with keys
            # "label", "score", "box", and "centroid"
            detected_obj = self._obj_detection(frame)
            self._append_frame(detected_obj)

            if visualize:
                self._visualize_frame(frame, detected_obj)

            for i in range(self.window_size):
                success, frame = self.datastream.step()
                if not success:
                    break

                detected_obj = self._obj_detection(frame)
                detected_obj = self._compute_motion(
                    self.frames[-1], detected_obj)
                self._append_frame(detected_obj)

                if visualize:
                    self._visualize_frame(frame, detected_obj)

            # generate commentary
            commentary = self.generate_commentary()
            print("Commentary:", commentary)

        if visualize:
            cv2.destroyAllWindows()
