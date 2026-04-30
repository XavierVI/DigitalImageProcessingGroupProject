"""Combined detector using both DETR (vehicle) and YOLO (traffic signs) models."""

import os
from typing import List, Dict, Optional
import numpy as np
import torch
from PIL import Image

from .object_detector import ObjectDetector


class CombinedDetector:
    """Combines vehicle detection (DETR) with traffic sign detection (YOLO).
    
    Runs both models on each frame and merges detections, assigning each
    detection a 'detection_type' to distinguish between vehicles and signs.
    """
    
    def __init__(
        self,
        vehicle_model: str = "PekingU/rtdetr_r50vd",
        vehicle_yolo_weights: Optional[str] = None,
        sign_model: str = "yolo",
        sign_yolo_weights: Optional[str] = None,
        device=None,
    ):
        """Initialize combined detector.
        
        Args:
            vehicle_model: Model for vehicle detection (DETR). Use "yolo" for YOLO.
            vehicle_yolo_weights: Path to YOLO weights for vehicle detection (if using YOLO)
            sign_model: Model for traffic sign detection. Typically "yolo".
            sign_yolo_weights: Path to YOLO weights for traffic sign detection
            device: torch device (cuda or cpu)
        """
        self.device = device or torch.device("cpu")
        
        # Vehicle detector (DETR or YOLO for general objects)
        self.vehicle_detector = ObjectDetector(
            vehicle_model,
            device=self.device,
            yolo_weights_path=vehicle_yolo_weights
        )
        
        # Traffic sign detector (YOLO fine-tuned for signs)
        self.sign_detector = ObjectDetector(
            sign_model,
            device=self.device,
            yolo_weights_path=sign_yolo_weights
        )
    
    def detect(self, frame, vehicle_threshold: float = 0.9, sign_threshold: float = 0.5) -> List[Dict]:
        """Detect both vehicles and traffic signs in the frame.
        
        Args:
            frame: Input image (numpy array or PIL Image)
            vehicle_threshold: Confidence threshold for vehicle detection
            sign_threshold: Confidence threshold for sign detection
            
        Returns:
            List of detected objects with 'detection_type' field ('vehicle' or 'traffic_sign')
        """
        # Run both detectors
        vehicle_detections = self.vehicle_detector.detect(frame, threshold=vehicle_threshold)
        sign_detections = self.sign_detector.detect(frame, threshold=sign_threshold)
        
        # Tag detections with their type
        for det in vehicle_detections:
            det['detection_type'] = 'vehicle'
        
        for det in sign_detections:
            det['detection_type'] = 'traffic_sign'
        
        # Merge results
        all_detections = vehicle_detections + sign_detections
        
        return all_detections
    
    def detect_separate(self, frame, vehicle_threshold: float = 0.9, sign_threshold: float = 0.5) -> tuple:
        """Detect vehicles and traffic signs separately.
        
        Useful if you want to process them differently.
        
        Args:
            frame: Input image (numpy array or PIL Image)
            vehicle_threshold: Confidence threshold for vehicle detection
            sign_threshold: Confidence threshold for sign detection
            
        Returns:
            Tuple of (vehicle_detections, sign_detections)
        """
        vehicle_detections = self.vehicle_detector.detect(frame, threshold=vehicle_threshold)
        sign_detections = self.sign_detector.detect(frame, threshold=sign_threshold)
        
        return vehicle_detections, sign_detections
