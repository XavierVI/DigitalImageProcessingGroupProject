"""Construct prompts for the LLM from object detection and motion data."""

from typing import List, Dict, Any
import json


class PromptConstructor:
    def __init__(self, keywords: List[str]):
        self.keywords = keywords
        self.base_instruction = (
            "You are a driving assistant for dashcam footage. "
            "Respond quickly, directly, and only with useful safety information. "
            "Prioritize immediate hazards over background details."
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
        sections = [
            self.base_instruction,
            "Output format: 1 to 3 short bullet points.",
            "If there is no immediate hazard, say 'No immediate hazard detected.'",
            "Mention what is happening, where it is, and what the driver should do.",
            "Do not copy raw detection text, confidence percentages, or timestep lines.",
            "Do not output phrases like 'number of objects', 'Detected', or 'Timestep'.",
        ]

        for i, obj in enumerate(frames):
            new_section = [f"Timestep {t + i}:" ]
            num_objs, objs = self._format_detections(obj)
            new_section.append(f"- number of objects: {num_objs}")
            new_section.append(objs)

            if num_objs > 0:
                motion = self._format_motion(obj, [])
                if motion:
                    new_section.append(motion)

            sections.append("\n".join(new_section))

        sections.append(
            "FINAL TASK: Write the response now. Be specific, brief, and actionable."
        )

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

