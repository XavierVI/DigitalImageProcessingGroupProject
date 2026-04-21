"""Construct prompts for the LLM from object detection and motion data."""

import json
from typing import List, Dict


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

    def _build_driver_guidance_header(self) -> str:
        """Return the shared instruction block for driver-facing commentary."""
        return (
            f"You are an in-car {self.context} copilot. "
            f"Give short, practical driver guidance using only provided detections. "
            f"Prioritize safety-critical items first. "
            f"If a traffic light is detected but its color is not explicitly provided in the input, "
            f"say that the color cannot be determined from the current data instead of guessing. "
            f"Do not invent objects or states that are not in the detections."
        )

    @staticmethod
    def _summarize_scene(detected_objects: List[Dict]) -> Dict:
        """Create a compact scene summary for the prompt."""
        counts: Dict[str, int] = {}
        for obj in detected_objects:
            label = obj.get("label", "unknown")
            counts[label] = counts.get(label, 0) + 1

        priority_order = ["person", "traffic light", "stop sign", "truck", "car", "bus", "bicycle"]
        unique_labels = list(counts.keys())
        unique_labels.sort(key=lambda lbl: (priority_order.index(lbl) if lbl in priority_order else 999, lbl))

        return {
            "total_detections": len(detected_objects),
            "label_counts": counts,
            "priority_labels": unique_labels[:6],
        }

    @staticmethod
    def _format_detections(detected_objects: List[Dict]) -> str:
        """Format detections as JSON for easier LLM parsing."""
        payload = []
        for obj in detected_objects[:6]:
            item = {
                "label": obj.get("label", "unknown"),
                "confidence": obj.get("score", "unknown"),
            }
            if "color" in obj:
                item["color"] = obj["color"]
            payload.append(item)
        return json.dumps(payload, indent=2)

    def construct_from_objects(self, detected_objects: List[Dict]) -> str:
        """Construct a prompt from detected objects.

        Args:
            detected_objects: List of detected objects with labels, scores, boxes

        Returns:
            str: Formatted prompt for the LLM
        """
        if not detected_objects:
            return (
                self._build_driver_guidance_header()
                + "\n\nNo objects were detected in the current scene. "
                + "Respond briefly that the road appears clear based on the available data."
            )

        scene_summary = self._summarize_scene(detected_objects)
        object_desc = self._format_detections(detected_objects)

        return (
            self._build_driver_guidance_header()
            + "\n\nScene summary JSON:\n"
            + json.dumps(scene_summary, indent=2)
            + "\n\nTop detections JSON (source of truth):\n"
            + "Treat this JSON as the source of truth:\n"
            + object_desc
            + "\n\nRequired response format:"
            + "\n1. Road overview (one short sentence)."
            + "\n2. Traffic controls ahead (sign/light and light color if known; otherwise unknown)."
            + "\n3. Immediate action (one short sentence)."
            + "\nKeep the full answer under 60 words."
        )

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
            return (
                self._build_driver_guidance_header()
                + "\n\nNo objects were detected in the current scene. "
                + "State that no immediate road objects are visible based on the current frame."
            )

        prompt = (
            self._build_driver_guidance_header()
            + "\n\nAnalyze the scene and motion data to provide driver alerts. "
            + "Use motion to highlight objects that may require caution. "
        )

        # Add object information
        for i, obj in enumerate(detected_objects[:5]):
            if i < len(motion_data):
                motion = motion_data[i]
                prompt += (
                    f"\n- {obj['label']} (confidence: {obj['score']}) "
                    f"at position {obj['box']}, moving with velocity {motion.get('velocity', 'unknown')}, "
                    f"speed {motion.get('speed', 'unknown')}. "
                )
            else:
                prompt += f"\n- {obj['label']} (confidence: {obj['score']}) at position {obj['box']}. "

        prompt += (
            "\n\nProvide a short driver-facing summary with the same required format: "
            "road overview, traffic signs/lights, traffic-light color if known, and immediate action."
        )

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
        prompt = (
            self._build_driver_guidance_header()
            + "\n\nAnalyze the temporal motion of objects and provide driver safety insights.\n"
        )

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
