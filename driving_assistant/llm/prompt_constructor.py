"""Construct prompts for the LLM from object detection and motion data."""

from typing import List, Dict, Any
import json


class PromptConstructor:
    """Handles serialization of detection and motion data into LLM prompts.

    This class converts spatiotemporal object information into structured prompts
    suitable for feeding to a Large Language Model to generate driver alerts.
    """

    def __init__(self, context: str = "driver assistance"):
        """Initialize the prompt constructor.

        Args:
            context (str): Context string for the prompts (e.g., 'driver assistance')
        """
        self.context = context

    def construct_from_objects(self, detected_objects: List[Dict]) -> str:
        """Construct a prompt from detected objects.

        Args:
            detected_objects: List of detected objects with labels, scores, boxes

        Returns:
            str: Formatted prompt for the LLM
        """
        if not detected_objects:
            return "No objects detected in the current scene."

        object_list = ", ".join([obj["label"] for obj in detected_objects])
        object_desc = json.dumps(detected_objects[:5], indent=2)

        prompt = (
            f"You are an assistant for a {self.context} system. "
            f"Provide a professional summary of the scene. "
            f"Detected objects: {object_list}. "
            f"Details: {object_desc}."
        )

        return prompt

    def construct_from_motion(
        self,
        detected_objects: List[Dict],
        motion_data: List[Dict]
    ) -> str:
        """Construct a prompt from detected objects and their motion.

        Args:
            detected_objects: List of detected objects
            motion_data: List of motion information (centroids, velocities, etc.)

        Returns:
            str: Formatted prompt including motion information
        """
        if not detected_objects:
            return "No objects detected in the current scene."

        prompt = (
            f"You are an assistant for a {self.context} system. "
            f"Analyze the scene and motion data to provide alerts. "
        )

        # Add object information
        for i, obj in enumerate(detected_objects[:5]):
            if i < len(motion_data):
                motion = motion_data[i]
                prompt += (
                    f"\n- {obj['label']} (confidence: {obj['score']}) "
                    f"at position {obj['box']}, moving with velocity {motion.get('velocity', 'unknown')}. "
                )
            else:
                prompt += f"\n- {obj['label']} (confidence: {obj['score']}) at position {obj['box']}. "

        prompt += "\nProvide any necessary alerts or warnings for the driver."

        return prompt

    def construct_from_timeseries(
        self,
        object_tracks: Dict[str, List[Dict]]
    ) -> str:
        """Construct a prompt from temporal object tracking data.

        Args:
            object_tracks: Dictionary mapping object IDs to their history over time

        Returns:
            str: Formatted prompt with temporal information
        """
        prompt = f"You are an assistant for a {self.context} system. "\
                 f"Analyze the temporal motion of objects and provide insights.\n"

        for obj_id, track in object_tracks.items():
            if track:
                # Get first and last positions
                first_pos = track[0].get("position", [0, 0])
                last_pos = track[-1].get("position", [0, 0])
                displacement = (
                    last_pos[0] - first_pos[0],
                    last_pos[1] - first_pos[1]
                )

                prompt += (
                    f"\nObject {obj_id}: "
                    f"moved from {first_pos} to {last_pos} "
                    f"(displacement: {displacement}). "
                    f"Track length: {len(track)} frames."
                )

        return prompt
