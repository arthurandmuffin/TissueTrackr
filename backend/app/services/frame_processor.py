import cv2
import numpy as np
from typing import List, Dict, Tuple

class FrameProcessor:
    """Service for processing video frames and detecting sharp features as anchor points."""
    
    def __init__(self, max_features: int = 500):
        """
        Initialize the frame processor.
        
        Args:
            max_features: Maximum number of features to detect per frame
        """
        self.max_features = max_features
        # Initialize SIFT detector for scale-invariant feature detection
        self.detector = cv2.SIFT_create(nfeatures=max_features)
    
    def detect_anchor_points(self, frame: np.ndarray) -> Dict:
        """
        Detect sharp features (anchor points) in a frame using SIFT.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Dictionary containing:
                - keypoints: List of detected keypoints
                - descriptors: Feature descriptors
                - count: Number of keypoints found
                - coordinates: List of (x, y) coordinates
        """
        if frame is None or frame.size == 0:
            return {
                "keypoints": [],
                "descriptors": None,
                "count": 0,
                "coordinates": []
            }
        
        # Convert to grayscale for feature detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect keypoints and descriptors
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        
        # Extract coordinates from keypoints
        coordinates = [(int(kp.pt[0]), int(kp.pt[1])) for kp in keypoints]
        
        return {
            "keypoints": keypoints,
            "descriptors": descriptors,
            "count": len(keypoints),
            "coordinates": coordinates,
            "sizes": [kp.size for kp in keypoints],
            "angles": [kp.angle for kp in keypoints]
        }
    
    def draw_anchor_points(self, frame: np.ndarray, anchor_points: Dict) -> np.ndarray:
        """
        Draw detected anchor points on the frame.
        
        Args:
            frame: Input frame
            anchor_points: Dictionary from detect_anchor_points
            
        Returns:
            Frame with drawn keypoints
        """
        output_frame = frame.copy()
        keypoints = anchor_points.get("keypoints", [])
        
        # Draw circles at each keypoint location
        for keypoint in keypoints:
            x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
            size = int(keypoint.size)
            cv2.circle(output_frame, (x, y), size // 2, (0, 255, 0), 2)
            cv2.circle(output_frame, (x, y), 2, (0, 0, 255), -1)
        
        # Add text with count
        cv2.putText(
            output_frame,
            f"Anchor Points: {anchor_points['count']}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        
        return output_frame

    def draw_landmarks(self, frame: np.ndarray, landmarks: List) -> np.ndarray:
        """
        Draw landmark points on the frame.

        Args:
            frame: Input frame
            landmarks: List of Landmark models

        Returns:
            Frame with landmarks drawn
        """
        output_frame = frame.copy()
        for landmark in landmarks:
            x, y = int(landmark.x), int(landmark.y)
            cv2.circle(output_frame, (x, y), 4, (0, 255, 0), 2)
            cv2.circle(output_frame, (x, y), 2, (0, 0, 255), -1)

        cv2.putText(
            output_frame,
            f"Landmarks: {len(landmarks)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        return output_frame
    
    def process_frame(self, frame: np.ndarray, draw: bool = False) -> Dict:
        """
        Process a single frame and detect anchor points.
        
        Args:
            frame: Input frame as numpy array
            draw: Whether to draw keypoints on the frame
            
        Returns:
            Dictionary with processing results
        """
        anchor_points = self.detect_anchor_points(frame)
        
        result = {
            "success": True,
            "anchor_points": anchor_points,
            "frame": None
        }
        
        if draw:
            result["frame"] = self.draw_anchor_points(frame, anchor_points)
        
        return result
