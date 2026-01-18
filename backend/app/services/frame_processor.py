import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional

from ..models import AnnotationRecord, GeometryType, Point2D, TransformMatrix

class FrameProcessor:
    """Service for processing video frames and detecting sharp features as anchor points."""

    def __init__(self, max_features: int = 500, max_dimension: Optional[int] = None):
        """
        Initialize the frame processor.
        
        Args:
            max_features: Maximum number of features to detect per frame
            max_dimension: Optional max length for width/height used in detection
        """
        self.max_features = max_features
        self.max_dimension = max_dimension
        # Initialize ORB detector for fast binary feature detection.
        self.detector = cv2.ORB_create(nfeatures=max_features)
    
    def detect_anchor_points(self, frame: np.ndarray) -> Dict:
        """
        Detect sharp features (anchor points) in a frame using ORB.
        
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

        scale = 1.0
        if self.max_dimension and self.max_dimension > 0:
            height, width = gray.shape[:2]
            max_side = max(height, width)
            if max_side > self.max_dimension:
                scale = self.max_dimension / max_side
                new_w = max(1, int(round(width * scale)))
                new_h = max(1, int(round(height * scale)))
                gray = cv2.resize(
                    gray, (new_w, new_h), interpolation=cv2.INTER_AREA
                )
        
        # Detect keypoints and descriptors
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)

        if scale != 1.0 and keypoints:
            inv_scale = 1.0 / scale
            scaled_keypoints = []
            for kp in keypoints:
                scaled_keypoints.append(
                    cv2.KeyPoint(
                        kp.pt[0] * inv_scale,
                        kp.pt[1] * inv_scale,
                        kp.size * inv_scale,
                        kp.angle,
                        kp.response,
                        kp.octave,
                        kp.class_id,
                    )
                )
            keypoints = scaled_keypoints

        if len(keypoints) > self.max_features:
            order = np.argsort([kp.response for kp in keypoints])[::-1]
            order = order[: self.max_features]
            keypoints = [keypoints[idx] for idx in order]
            if descriptors is not None:
                descriptors = descriptors[order]
        
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

    def draw_annotations(
        self, frame: np.ndarray, annotations: List[AnnotationRecord]
    ) -> np.ndarray:
        """
        Draw annotation geometries on the frame.

        Args:
            frame: Input frame
            annotations: List of AnnotationRecord models

        Returns:
            Frame with annotations drawn
        """
        if not annotations:
            return frame

        output_frame = frame.copy()
        for annotation in annotations:
            points = annotation.points or []
            if not points:
                continue

            transform = annotation.local_transform or annotation.global_transform
            coords = self._apply_transform(points, transform)
            if coords.size == 0:
                continue

            color = self._annotation_color(annotation.metadata)
            thickness = 2
            if annotation.geometry_type == GeometryType.point:
                x, y = int(coords[0][0]), int(coords[0][1])
                cv2.circle(output_frame, (x, y), 6, color, -1)
            elif annotation.geometry_type == GeometryType.bbox and len(coords) >= 2:
                x0, y0 = coords[0]
                x1, y1 = coords[1]
                left, right = int(min(x0, x1)), int(max(x0, x1))
                top, bottom = int(min(y0, y1)), int(max(y0, y1))
                cv2.rectangle(output_frame, (left, top), (right, bottom), color, thickness)
            elif annotation.geometry_type in (GeometryType.polyline, GeometryType.polygon):
                if len(coords) < 2:
                    x, y = int(coords[0][0]), int(coords[0][1])
                    cv2.circle(output_frame, (x, y), 6, color, -1)
                else:
                    poly = coords.reshape((-1, 1, 2)).astype(np.int32)
                    closed = annotation.geometry_type == GeometryType.polygon
                    cv2.polylines(output_frame, [poly], closed, color, thickness)
            else:
                x, y = int(coords[0][0]), int(coords[0][1])
                cv2.circle(output_frame, (x, y), 6, color, -1)

        return output_frame

    def _annotation_color(self, metadata: Dict) -> Tuple[int, int, int]:
        raw = metadata.get("color") if metadata else None
        if not isinstance(raw, str):
            return (255, 140, 0)
        value = raw.strip()
        if value.startswith("#"):
            value = value[1:]
        if len(value) != 6:
            return (255, 140, 0)
        try:
            r = int(value[0:2], 16)
            g = int(value[2:4], 16)
            b = int(value[4:6], 16)
        except ValueError:
            return (255, 140, 0)
        return (b, g, r)

    def _apply_transform(
        self,
        points: List[Point2D],
        transform: Optional[TransformMatrix],
    ) -> np.ndarray:
        coords = np.array([[pt.x, pt.y] for pt in points], dtype=np.float32)
        if coords.size == 0 or transform is None:
            return coords

        matrix = np.array(transform.matrix, dtype=np.float32)
        if matrix.shape == (2, 3):
            coords = coords.reshape(-1, 1, 2)
            transformed = cv2.transform(coords, matrix)
            return transformed.reshape(-1, 2)
        if matrix.shape == (3, 3):
            coords = coords.reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(coords, matrix)
            return transformed.reshape(-1, 2)

        return coords
    
    def draw_stats(
        self,
        frame: np.ndarray,
        new_landmarks: bool,
        total_landmarks: int,
        origin: Tuple[int, int] = (10, 55),
    ) -> np.ndarray:
        output_frame = frame.copy()
        x0, y0 = origin
        cv2.putText(
            output_frame,
            f"New Landmarks: {new_landmarks}",
            (x0, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            output_frame,
            f"Map Landmarks: {total_landmarks}",
            (x0, y0 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
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
                str(landmark.id),
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                1,
            )

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
