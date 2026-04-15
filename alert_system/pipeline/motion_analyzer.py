"""Analyze motion of detected objects across frames."""

from typing import List, Dict, Tuple, Optional
import numpy as np


class MotionAnalyzer:
    """Analyzes motion of detected objects across multiple frames.

    Computes centroids and velocities from bounding boxes.
    """

    @staticmethod
    def compute_centroid(box: List[float]) -> Tuple[float, float]:
        """Compute the centroid of a bounding box.

        Args:
            box (list): Bounding box as [xmin, ymin, xmax, ymax]

        Returns:
            tuple: (center_x, center_y) coordinates
        """
        xmin, ymin, xmax, ymax = box
        center_x = (xmax + xmin) / 2
        center_y = (ymax + ymin) / 2
        return center_x, center_y

    @staticmethod
    def compute_velocity(
        prev_centroid: Tuple[float, float],
        curr_centroid: Tuple[float, float],
        time_delta: float = 1.0
    ) -> Tuple[float, float]:
        """Compute velocity from two consecutive centroids.

        Args:
            prev_centroid (tuple): Previous (x, y) position
            curr_centroid (tuple): Current (x, y) position
            time_delta (float): Time between frames (default: 1.0)

        Returns:
            tuple: (vx, vy) velocity components
        """
        vx = (curr_centroid[0] - prev_centroid[0]) / time_delta
        vy = (curr_centroid[1] - prev_centroid[1]) / time_delta
        return vx, vy

    @staticmethod
    def compute_speed(velocity: Tuple[float, float]) -> float:
        """Compute the magnitude of velocity.

        Args:
            velocity (tuple): (vx, vy) velocity components

        Returns:
            float: Speed magnitude
        """
        vx, vy = velocity
        return np.sqrt(vx**2 + vy**2)

    @classmethod
    def compute_motion_from_objects(
        cls,
        prev_objects: List[Dict],
        curr_objects: List[Dict],
        time_delta: float = 1.0
    ) -> List[Dict]:
        """Compute motion data between two frames of detections.

        Args:
            prev_objects: Previous frame's detected objects
            curr_objects: Current frame's detected objects
            time_delta: Time between frames

        Returns:
            list: List of motion data for matched objects
        """
        motion_data = []

        # Simple matching: assume objects are in same order or use first N matches
        for i, curr_obj in enumerate(curr_objects):
            if i >= len(prev_objects):
                # No previous detection for this object
                motion_data.append({
                    "label": curr_obj["label"],
                    "current_box": curr_obj["box"],
                    "current_centroid": cls.compute_centroid(curr_obj["box"]),
                    "velocity": (0, 0),
                    "speed": 0.0,
                    "matched": False
                })
                continue

            prev_obj = prev_objects[i]

            # Assume objects have similar labels for matching
            if curr_obj["label"] != prev_obj["label"]:
                motion_data.append({
                    "label": curr_obj["label"],
                    "current_box": curr_obj["box"],
                    "current_centroid": cls.compute_centroid(curr_obj["box"]),
                    "velocity": (0, 0),
                    "speed": 0.0,
                    "matched": False
                })
                continue

            prev_centroid = cls.compute_centroid(prev_obj["box"])
            curr_centroid = cls.compute_centroid(curr_obj["box"])
            velocity = cls.compute_velocity(prev_centroid, curr_centroid, time_delta)
            speed = cls.compute_speed(velocity)

            motion_data.append({
                "label": curr_obj["label"],
                "previous_box": prev_obj["box"],
                "previous_centroid": prev_centroid,
                "current_box": curr_obj["box"],
                "current_centroid": curr_centroid,
                "velocity": velocity,
                "speed": speed,
                "matched": True
            })

        return motion_data

    @classmethod
    def compute_acceleration(
        cls,
        prev_velocity: Tuple[float, float],
        curr_velocity: Tuple[float, float],
        time_delta: float = 1.0
    ) -> Tuple[float, float]:
        """Compute acceleration from two velocity measurements.

        Args:
            prev_velocity (tuple): Previous (vx, vy)
            curr_velocity (tuple): Current (vx, vy)
            time_delta (float): Time between measurements

        Returns:
            tuple: (ax, ay) acceleration components
        """
        ax = (curr_velocity[0] - prev_velocity[0]) / time_delta
        ay = (curr_velocity[1] - prev_velocity[1]) / time_delta
        return ax, ay
