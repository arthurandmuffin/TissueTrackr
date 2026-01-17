from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import cv2
import numpy as np

from ..models import (
    AnnotationCreate,
    AnnotationRecord,
    AnnotationsIn,
    FrameState,
    GlobalAnchor,
    GlobalAnchorHint,
    Landmark,
    LandmarkBinding,
    LocalAnchor,
    LocalAnchorHint,
    Point2D,
    TransformMatrix,
    TransformType,
)
from .frame_processor import FrameProcessor


@dataclass
class LocalAnchorState:
    keyframe_points: np.ndarray
    last_points: np.ndarray
    transform_type: TransformType
    patch_center: Tuple[float, float]
    patch_radius: int


@dataclass
class GlobalAnchorState:
    landmark_ids: List[str]
    keyframe_points: np.ndarray
    transform_type: TransformType


@dataclass
class AnnotationState:
    record: AnnotationRecord
    local_state: Optional[LocalAnchorState]
    global_state: Optional[GlobalAnchorState]


class LandmarkMap:
    def __init__(self, max_features: int = 500):
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        self.next_id = 1
        self.prev_descriptors: Optional[np.ndarray] = None
        self.prev_ids: List[str] = []
        self.track_ages: Dict[str, int] = {}
        self.max_features = max_features

    def _new_id(self) -> str:
        value = f"lm_{self.next_id}"
        self.next_id += 1
        return value

    def update(
        self,
        keypoints: List[cv2.KeyPoint],
        descriptors: Optional[np.ndarray],
    ) -> Tuple[List[Landmark], Dict[str, Tuple[float, float]]]:
        if descriptors is None or len(keypoints) == 0:
            self.prev_descriptors = descriptors
            self.prev_ids = []
            self.track_ages = {}
            return [], {}

        if self.prev_descriptors is None or len(self.prev_ids) == 0:
            ids = [self._new_id() for _ in keypoints]
        else:
            matches = self.matcher.match(self.prev_descriptors, descriptors)
            ids = [None] * len(keypoints)
            used_prev = set()
            for match in matches:
                if match.queryIdx >= len(self.prev_ids):
                    continue
                prev_id = self.prev_ids[match.queryIdx]
                if prev_id in used_prev:
                    continue
                ids[match.trainIdx] = prev_id
                used_prev.add(prev_id)
            for idx, lm_id in enumerate(ids):
                if lm_id is None:
                    ids[idx] = self._new_id()

        new_ages: Dict[str, int] = {}
        landmarks: List[Landmark] = []
        positions: Dict[str, Tuple[float, float]] = {}
        for idx, kp in enumerate(keypoints):
            lm_id = ids[idx]
            age = self.track_ages.get(lm_id, 0) + 1
            new_ages[lm_id] = age
            x, y = float(kp.pt[0]), float(kp.pt[1])
            landmarks.append(
                Landmark(
                    id=lm_id,
                    x=x,
                    y=y,
                    size=float(kp.size),
                    angle=float(kp.angle),
                    age=age,
                )
            )
            positions[lm_id] = (x, y)

        self.track_ages = new_ages
        self.prev_descriptors = descriptors
        self.prev_ids = ids
        return landmarks, positions


class AnchorManager:
    def __init__(
        self,
        max_features: int = 500,
        local_patch_radius: int = 96,
        local_min_points: int = 20,
        local_max_points: int = 80,
        global_k_nearest: int = 12,
        min_landmark_age: int = 2,
    ):
        self.frame_processor = FrameProcessor(max_features=max_features)
        self.landmark_map = LandmarkMap(max_features=max_features)
        self.local_patch_radius = local_patch_radius
        self.local_min_points = local_min_points
        self.local_max_points = local_max_points
        self.global_k_nearest = global_k_nearest
        self.min_landmark_age = min_landmark_age
        self.frame_index = 0
        self.prev_gray: Optional[np.ndarray] = None
        self.last_frame: Optional[np.ndarray] = None
        self.last_landmarks: List[Landmark] = []
        self.annotations: Dict[str, AnnotationState] = {}

    def render_frame(self, frame: np.ndarray, landmarks: List[Landmark]) -> np.ndarray:
        return self.frame_processor.draw_landmarks(frame, landmarks)

    def process_frame(self, frame: np.ndarray) -> FrameState:
        self.frame_index += 1
        frame_id = f"frame-{self.frame_index}"
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detection = self.frame_processor.detect_anchor_points(frame)
        landmarks, positions = self.landmark_map.update(
            detection["keypoints"], detection["descriptors"]
        )

        if self.prev_gray is not None:
            for state in self.annotations.values():
                if state.local_state is not None:
                    state.record.local_transform = self._update_local_anchor(
                        state.local_state, self.prev_gray, gray
                    )
                if state.global_state is not None:
                    state.record.global_transform = self._update_global_anchor(
                        state.global_state, positions
                    )
                state.record.updated_at = self._now()

        self.prev_gray = gray
        self.last_frame = frame
        self.last_landmarks = landmarks

        return FrameState(
            frame_id=frame_id,
            timestamp=self._now(),
            landmarks=landmarks,
            annotations=[state.record for state in self.annotations.values()],
        )

    def register_annotations(self, payload: AnnotationsIn) -> List[AnnotationRecord]:
        if self.last_frame is None:
            raise ValueError("No frame available for anchor binding yet.")

        frame_id = payload.frame_id or f"frame-{self.frame_index}"
        created: List[AnnotationRecord] = []
        for annotation in payload.annotations:
            record = self._build_annotation_record(frame_id, annotation)
            local_anchor, local_state = self._build_local_anchor(
                record, annotation.local_hint
            )
            global_anchor, global_state = self._build_global_anchor(
                record, annotation.global_hint
            )
            record.local_anchor = local_anchor
            record.global_anchor = global_anchor

            self.annotations[record.id] = AnnotationState(
                record=record, local_state=local_state, global_state=global_state
            )
            created.append(record)
        return created

    def _build_annotation_record(
        self, frame_id: str, annotation: AnnotationCreate
    ) -> AnnotationRecord:
        now = self._now()
        return AnnotationRecord(
            id=annotation.id or str(uuid4()),
            frame_id=frame_id,
            geometry_type=annotation.geometry_type,
            points=annotation.points,
            metadata=annotation.metadata,
            created_at=now,
            updated_at=now,
        )

    def _build_local_anchor(
        self, record: AnnotationRecord, hint: Optional[LocalAnchorHint]
    ) -> Tuple[Optional[LocalAnchor], Optional[LocalAnchorState]]:
        if self.last_frame is None or not record.points:
            return None, None

        transform_type = hint.transform_type if hint and hint.transform_type else TransformType.similarity
        radius = hint.patch_radius if hint and hint.patch_radius else self.local_patch_radius
        radius = int(max(64, min(128, radius)))

        center = self._centroid(record.points)
        gray = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY)
        x0, y0, x1, y1 = self._patch_bounds(center, radius, gray.shape)
        roi = gray[y0:y1, x0:x1]

        keyframe_points = self._detect_patch_points(roi, x0, y0)
        if keyframe_points.size == 0:
            keyframe_points = np.array([[center]], dtype=np.float32)

        local_anchor = LocalAnchor(
            patch_center=Point2D(x=center[0], y=center[1]),
            patch_radius=radius,
            keyframe_points=self._points_to_model(keyframe_points),
            transform_type=transform_type,
        )
        state = LocalAnchorState(
            keyframe_points=keyframe_points.copy(),
            last_points=keyframe_points.copy(),
            transform_type=transform_type,
            patch_center=center,
            patch_radius=radius,
        )
        return local_anchor, state

    def _build_global_anchor(
        self, record: AnnotationRecord, hint: Optional[GlobalAnchorHint]
    ) -> Tuple[Optional[GlobalAnchor], Optional[GlobalAnchorState]]:
        if not self.last_landmarks or not record.points:
            return None, None

        transform_type = hint.transform_type if hint and hint.transform_type else TransformType.similarity
        k_nearest = hint.k_nearest if hint and hint.k_nearest else self.global_k_nearest
        k_nearest = max(6, min(20, int(k_nearest)))

        center = self._centroid(record.points)
        candidates = [lm for lm in self.last_landmarks if lm.age >= self.min_landmark_age]
        if not candidates:
            candidates = self.last_landmarks

        sorted_landmarks = sorted(
            candidates,
            key=lambda lm: (lm.x - center[0]) ** 2 + (lm.y - center[1]) ** 2,
        )
        selected = sorted_landmarks[: max(1, int(k_nearest))]

        bindings = [
            LandmarkBinding(
                id=lm.id, keyframe_position=Point2D(x=lm.x, y=lm.y)
            )
            for lm in selected
        ]
        anchor = GlobalAnchor(landmarks=bindings, transform_type=transform_type)
        keyframe_points = np.array(
            [[lm.x, lm.y] for lm in selected], dtype=np.float32
        )
        state = GlobalAnchorState(
            landmark_ids=[lm.id for lm in selected],
            keyframe_points=keyframe_points,
            transform_type=transform_type,
        )
        return anchor, state

    def _update_local_anchor(
        self, state: LocalAnchorState, prev_gray: np.ndarray, gray: np.ndarray
    ) -> Optional[TransformMatrix]:
        if state.last_points.size == 0:
            return None

        new_points, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            gray,
            state.last_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
        )
        if new_points is None or status is None:
            return None

        mask = status.reshape(-1) == 1
        if mask.sum() < 4:
            state.last_points = state.last_points[mask]
            state.keyframe_points = state.keyframe_points[mask]
            return None

        state.last_points = new_points[mask]
        state.keyframe_points = state.keyframe_points[mask]

        return self._estimate_transform(
            state.keyframe_points.reshape(-1, 2),
            state.last_points.reshape(-1, 2),
            state.transform_type,
        )

    def _update_global_anchor(
        self, state: GlobalAnchorState, positions: Dict[str, Tuple[float, float]]
    ) -> Optional[TransformMatrix]:
        src_points = []
        dst_points = []
        for idx, lm_id in enumerate(state.landmark_ids):
            if lm_id not in positions:
                continue
            src_points.append(state.keyframe_points[idx])
            dst_points.append(positions[lm_id])

        if len(src_points) < 4:
            return None

        return self._estimate_transform(
            np.array(src_points, dtype=np.float32),
            np.array(dst_points, dtype=np.float32),
            state.transform_type,
        )

    def _estimate_transform(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray,
        transform_type: TransformType,
    ) -> Optional[TransformMatrix]:
        if transform_type == TransformType.homography:
            matrix, inliers = cv2.findHomography(
                src_points, dst_points, cv2.RANSAC, 3.0
            )
        elif transform_type == TransformType.affine:
            matrix, inliers = cv2.estimateAffine2D(
                src_points, dst_points, method=cv2.RANSAC, ransacReprojThreshold=3.0
            )
        else:
            matrix, inliers = cv2.estimateAffinePartial2D(
                src_points, dst_points, method=cv2.RANSAC, ransacReprojThreshold=3.0
            )

        if matrix is None:
            return None

        inlier_count = int(inliers.sum()) if inliers is not None else None
        return TransformMatrix(
            matrix=matrix.tolist(),
            inliers=inlier_count,
        )

    def _detect_patch_points(
        self, roi: np.ndarray, x_offset: int, y_offset: int
    ) -> np.ndarray:
        if roi.size == 0:
            return np.empty((0, 1, 2), dtype=np.float32)

        corners = cv2.goodFeaturesToTrack(
            roi,
            maxCorners=self.local_max_points,
            qualityLevel=0.01,
            minDistance=4,
            useHarrisDetector=False,
        )
        if corners is None:
            return np.empty((0, 1, 2), dtype=np.float32)

        points = corners.reshape(-1, 2)
        if len(points) > self.local_max_points:
            points = points[: self.local_max_points]
        points[:, 0] += x_offset
        points[:, 1] += y_offset
        return points.astype(np.float32).reshape(-1, 1, 2)

    def _centroid(self, points: List[Point2D]) -> Tuple[float, float]:
        xs = [pt.x for pt in points]
        ys = [pt.y for pt in points]
        return float(sum(xs) / len(xs)), float(sum(ys) / len(ys))

    def _patch_bounds(
        self, center: Tuple[float, float], radius: int, shape: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        height, width = shape[0], shape[1]
        x0 = max(0, int(center[0] - radius))
        y0 = max(0, int(center[1] - radius))
        x1 = min(width, int(center[0] + radius))
        y1 = min(height, int(center[1] + radius))
        return x0, y0, x1, y1

    def _points_to_model(self, points: np.ndarray) -> List[Point2D]:
        if points.size == 0:
            return []
        flat = points.reshape(-1, 2)
        return [Point2D(x=float(pt[0]), y=float(pt[1])) for pt in flat]

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)
