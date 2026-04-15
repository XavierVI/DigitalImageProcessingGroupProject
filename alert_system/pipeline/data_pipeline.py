"""End-to-end data pipeline combining object detection, motion analysis, and LLM commentary."""

from typing import List, Dict, Optional, Tuple
from PIL import Image

from traffic_pipeline.object_detection.object_detector import ObjectDetector
from traffic_pipeline.pipeline.motion_analyzer import MotionAnalyzer
from traffic_pipeline.llm.prompt_constructor import PromptConstructor
from traffic_pipeline.llm.commentary_generator import CommentaryGenerator


class DataPipeline:
    """End-to-end pipeline for processing images and generating driver alerts.

    Orchestrates object detection, motion analysis, prompt construction, and LLM inference.

    Args:
        object_detector: ObjectDetector instance
        motion_analyzer: MotionAnalyzer class/methods
        prompt_constructor: PromptConstructor instance
        commentary_generator: CommentaryGenerator instance
    """

    def __init__(
        self,
        object_detector: ObjectDetector,
        prompt_constructor: PromptConstructor,
        commentary_generator: CommentaryGenerator,
        detection_threshold: float = 0.9
    ):
        """Initialize the data pipeline.

        Args:
            object_detector: Detector for objects in images
            prompt_constructor: Constructs LLM prompts from detection data
            commentary_generator: Generates text from prompts
            detection_threshold: Confidence threshold for object detection
        """
        self.object_detector = object_detector
        self.motion_analyzer = MotionAnalyzer()
        self.prompt_constructor = prompt_constructor
        self.commentary_generator = commentary_generator
        self.detection_threshold = detection_threshold

        # History for motion tracking across frames
        self.prev_detections = None

    def process_frame(self, image: Image.Image) -> Dict:
        """Process a single frame: detect objects and generate commentary.

        Args:
            image (PIL.Image): Input image

        Returns:
            dict: Contains 'detections', 'motion', 'prompt', 'commentary'
        """
        # Step 1: Detect objects
        detection_results = self.object_detector.detect(image, self.detection_threshold)
        detections = detection_results["objects"]

        # Step 2: Compute motion if we have previous frame
        motion_data = []
        if self.prev_detections is not None:
            motion_data = self.motion_analyzer.compute_motion_from_objects(
                self.prev_detections,
                detections
            )

        # Step 3: Construct prompt
        if motion_data:
            prompt = self.prompt_constructor.construct_from_motion(detections, motion_data)
        else:
            prompt = self.prompt_constructor.construct_from_objects(detections)

        # Step 4: Generate commentary
        commentary = self.commentary_generator.generate(prompt)

        # Update history
        self.prev_detections = detections

        return {
            "detections": detections,
            "motion": motion_data,
            "prompt": prompt,
            "commentary": commentary
        }

    def process_frames_batch(self, images: List[Image.Image]) -> List[Dict]:
        """Process multiple frames sequentially.

        Args:
            images: List of PIL Images

        Returns:
            list: List of processing results
        """
        results = []
        for image in images:
            results.append(self.process_frame(image))
        return results

    def process_video_frames(
        self,
        frames: List,
        skip_frames: int = 1
    ) -> List[Dict]:
        """Process frames from a video.

        Args:
            frames: List of frames (numpy arrays in BGR format)
            skip_frames: Process only every nth frame

        Returns:
            list: List of processing results
        """
        results = []
        for idx, frame in enumerate(frames):
            if idx % skip_frames != 0:
                continue

            # Convert BGR to RGB
            import cv2
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame)

            results.append(self.process_frame(image))

        return results

    def reset_history(self):
        """Reset motion tracking history (useful for new video sequences)."""
        self.prev_detections = None

    def step(self):
        """Legacy method for compatibility with notebook code.

        Process a single image and serialize information for future use.
        """
        # This would be implemented with specific image source in actual usage
        pass

    def generate_commentary(self):
        """Legacy method for compatibility with notebook code.

        Generate commentary using time series data.
        """
        # This would aggregate time series data from previous steps
        pass
