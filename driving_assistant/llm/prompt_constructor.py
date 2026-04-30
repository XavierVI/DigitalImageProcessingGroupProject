"""Construct prompts for the LLM from object detection and motion data."""

from typing import List, Dict, Any
import json


class PromptConstructor:
    def __init__(self, keywords: List[str] = None):
        # self.base_instruction = (
        #     f"You are a professional driving assistant. "
        #     "Analyze the following data and provide concise, "
        #     "safety-critical alerts."
        # )

        self.system_instruction = (
            "You are a critical driving safety assistant. Analyze the temporal scene data "
            "and identify immediate collision risks or traffic violations.\n\n"
            "Rules:\n"
            "1. Output ONLY valid JSON.\n"
            "2. No markdown formatting, no explanations.\n"
            "3. Use format: {\"warning\": true/false, \"message\": \"<brief hazard>\"}\n\n"
            "Example Input:\n"
            "Timestep 0:\n- Pedestrian: close ahead, approaching rapidly\n\n"
            "Example Output:\n"
            "{\"warning\": true, \"message\": \"Pedestrian close ahead and approaching rapidly.\"}\n"
        )

        if keywords:
            self.system_instruction += (
                f"\nFocus on using these elements: {', '.join(keywords)}.\n"
            )

    def generate_prompt(self, frames: List[List[Dict]], t: int):
        """Main method to generate a prompt for the LLM.
        
        Args:
            frames: A list of frames over time. Each frame contains a list of detected objects.
            t: Starting timestep (for temporal context)
        Returns:
            A dictionary with 'system' and 'user' keys for the LLM input.
        """
        user_content = []

        for i, frame_objs in enumerate(frames):
            user_content.append(f"Timestep {t + i}:\n")

            if not frame_objs:
                user_content.append(" - No objects detected.")
                continue
            
            for obj in frame_objs:
                user_content.append(self._format_detections(obj))

        return {
            "system": self.system_instruction,
            "user": "\n".join(user_content)
        }

    def _format_detections(self, objs: Dict) -> str:
        label = objs.get('label', 'unknown')
        centroid = objs.get('centroid')
        velocity = objs.get('velocity', (0, 0))
        area = objs.get('area', 0)

        # spatial evaluation
        if centroid:
            cx, cy = centroid
            position = "close ahead" if cy > 240 and area > 5000 else "far ahead"
            side = "left" if cx < 320 else "right"
            loc_str = f"{position}, {side} side"
        else:
            loc_str = "unknown location"

        # Kinematic evaluation (CPU-side)
        vx, vy = velocity
        speed = (vx ** 2 + vy ** 2) ** 0.5

        if speed > 15:
            motion_str = "approaching rapidly" if vy > 0 else "receding rapidly"
        elif speed > 5:
            motion_str = "approaching" if vy > 0 else "receding"
        else:
            motion_str = "static"

        # Dense, token-efficient string
        return f"  - {label}: {loc_str}, {motion_str}"

    def _format_motion(self, objs: List[Dict]) -> str:
        lines = ["Motion:"]

        for obj in objs[:5]:
            vx, vy = obj.get('velocity', (0, 0))
            speed = (vx ** 2 + vy ** 2) ** 0.5
            if speed > 5:  # only report meaningful motion
                direction = "approaching" if vy > 0 else "moving away"
                lines.append(
                    f"  - {obj['label']} is {direction} "
                    f"(speed ~{speed:.0f} px/frame)"
                )

        return "\n".join(lines) if len(lines) > 1 else ""

