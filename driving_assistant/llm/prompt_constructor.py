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
        
        labels = [o['label'] for o in objs]
        
        return len(labels), f"- Objects in view: {', '.join(labels)}."

    def _format_motion(self, objs: List[Dict], motion: List[Dict]) -> str:
        output = "Kinematic Data:"
        
        for i, obj in enumerate(objs[:5]):  # Keep it concise for tokens
            vel = motion[i].get('velocity', (0, 0)) if i < len(
                motion) else (0, 0)
        
            output += f"\n- {obj['label']}: Box {obj['box']}, Velocity {vel}"
        
        return output

