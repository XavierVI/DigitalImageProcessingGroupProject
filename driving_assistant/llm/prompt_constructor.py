"""Construct prompts for the LLM from object detection and motion data."""

from typing import List, Dict, Any
import json


class PromptConstructor:
    def __init__(self, keywords: List[str]):
        self.keywords = keywords
        # Define a consistent system instruction
        self.base_instruction = (
            f"You are a professional driving assistant. "
            "Analyze the following data and provide concise, "
            "safety-critical alerts."
        )
        self.base_instruction += (
            f"\nFocus on these keywords: {', '.join(self.keywords)}."
        )

    def generate_prompt(self, frames, t) -> str:
        """Main method to generate a prompt for the LLM.
        
        Args:
            frames: A list of detected objects over time. Each object is a dictionary
                containing the keys: label, score, vel, centroid.
            t: Current timestep (for temporal context)
        Returns:
            A formatted string prompt for the LLM.
        """
        sections = [self.base_instruction]

        for i, obj in enumerate(frames):
            new_section = f"Timestep {t + i}: ```"
            # extract relevant data for the current frame

            num_objs, objs = self._format_detections(obj)
            new_section += f"- number of objects: {num_objs}\n{objs}"

            if num_objs > 0:
                new_section += "\n" + self._format_motion(obj, [])

        # Add the 'Ask' at the end
        sections.append(
            "\nFINAL TASK: Provide a concise summary and any urgent hazard warnings.")

        return "\n".join(sections)

    def _format_detections(self, objs: List[Dict]) -> tuple[int, str]:
        if len(objs) == 0:
            return 0, "Scene Status: No objects detected."

        lines = []
        for o in objs:
            centroid = o.get('centroid', None)
            if centroid:
                cx, cy = centroid
                # Assume standard dashcam resolution ~480 height, ~640 width
                position = "close ahead" if cy > 240 else "far ahead"
                side = "left" if cx < 320 else "right"
                lines.append(
                    f"  - {o['label']} ({o['score']:.0%} conf) "
                    f"— {position}, {side} side"
                )
            else:
                lines.append(f"  - {o['label']} ({o['score']:.0%} conf)")

        return len(lines), "Detected:\n" + "\n".join(lines)

    def _format_motion(self, objs: List[Dict], motion: List[Dict]) -> str:
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

