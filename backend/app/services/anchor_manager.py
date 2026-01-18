from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple
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


@dataclass
class MapLandmark:
    id: str
    position: Tuple[float, float]
    descriptor: np.ndarray
    age: int = 1
    last_seen_frame: int = 0


@dataclass
class MapMatch:
    transform: np.ndarray
    inlier_matches: List[cv2.DMatch]
    map_ids: List[str]

@dataclass
class MapState:
    landmarks: Dict[str, MapLandmark]
    next_landmark_id: int = 1

    def new_landmark_id(self) -> str:
        value = f"map_lm_{self.next_landmark_id}"
        self.next_landmark_id += 1
        return value


class AnchorManager:
    def __init__(
        self,
        max_features: int = 3000,
        max_detection_dimension: Optional[int] = None,
        local_patch_radius: int = 96,
        local_min_points: int = 20,
        local_max_points: int = 80,
        global_k_nearest: int = 12,
        min_landmark_age: int = 2,
        render_landmarks: bool = True,
        map_max_landmarks: int = 5000,
        map_max_new_per_frame: int = 50,
        map_min_inliers: int = 10,
        map_lost_threshold: int = 5,
        map_match_ratio: float = 0.85,
        allow_new_landmarks: bool = True,
        freeze_new_landmarks_after_frames: Optional[int] = 600,
    ):
        self.frame_processor = FrameProcessor(
            max_features=max_features,
            max_dimension=max_detection_dimension,
        )
        self.local_patch_radius = local_patch_radius
        self.local_min_points = local_min_points
        self.local_max_points = local_max_points
        self.global_k_nearest = global_k_nearest
        self.min_landmark_age = min_landmark_age
        self.render_landmarks = render_landmarks
        self.map_max_landmarks = map_max_landmarks
        self.map_max_new_per_frame = map_max_new_per_frame
        self.map_min_inliers = map_min_inliers
        self.map_lost_threshold = map_lost_threshold
        self.map_match_ratio = map_match_ratio
        self.allow_new_landmarks = allow_new_landmarks
        self.freeze_new_landmarks_after_frames = freeze_new_landmarks_after_frames
        self.map_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.map_state: Optional[MapState] = None
        self.lost_frames = 0
        self.last_map_transform: Optional[np.ndarray] = None
        self.frame_index = 0
        self.prev_gray: Optional[np.ndarray] = None
        self.last_frame: Optional[np.ndarray] = None
        self.last_landmarks: List[Landmark] = []
        self.annotations: Dict[str, AnnotationState] = {}

    def render_frame(
        self,
        frame: np.ndarray,
        landmarks: List[Landmark],
        annotations: Optional[List[AnnotationRecord]] = None,
    ) -> np.ndarray:
        output_frame = frame.copy()
        total_landmarks = 0
        if self.map_state is not None:
            total_landmarks = len(self.map_state.landmarks)
        output_frame = self.frame_processor.draw_stats(
            output_frame,
            self._should_allow_new_landmarks(),
            total_landmarks,
        )
        if self.render_landmarks:
            output_frame = self.frame_processor.draw_landmarks(
                output_frame, landmarks
            )
        if annotations:
            output_frame = self.frame_processor.draw_annotations(
                output_frame, annotations
            )
        return output_frame

    def process_frame(self, frame: np.ndarray) -> FrameState:
        self.frame_index += 1
        frame_id = f"frame-{self.frame_index}"
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detection = self.frame_processor.detect_anchor_points(frame)
        landmarks, positions = self._update_global_map(
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

    def set_allow_new_landmarks(self, enabled: bool) -> None:
        self.allow_new_landmarks = bool(enabled)

    def _should_allow_new_landmarks(self) -> bool:
        if not self.allow_new_landmarks:
            return False
        if (
            self.freeze_new_landmarks_after_frames is not None
            and len(self.map_state.landmarks) >= self.map_max_landmarks
        ):
            return False
        return True

    def _update_global_map(
        self,
        keypoints: List[cv2.KeyPoint],
        descriptors: Optional[np.ndarray],
    ) -> Tuple[List[Landmark], Dict[str, Tuple[float, float]]]:
        if descriptors is None or len(keypoints) == 0:
            self.lost_frames = min(self.map_lost_threshold, self.lost_frames + 1)
            return [], {}

        if self.map_state is None or not self.map_state.landmarks:
            return self._initialize_map(keypoints, descriptors)

        match = self._match_map(self.map_state, keypoints, descriptors)
        if match is None:
            self.lost_frames = min(self.map_lost_threshold, self.lost_frames + 1)
            return [], {}
        else:
            self.lost_frames = 0

        return self._update_map_from_match(
            self.map_state, match, keypoints, descriptors
        )

    def _initialize_map(
        self,
        keypoints: List[cv2.KeyPoint],
        descriptors: np.ndarray,
    ) -> Tuple[List[Landmark], Dict[str, Tuple[float, float]]]:
        self.map_state = MapState(landmarks={})

        landmarks: List[Landmark] = []
        positions: Dict[str, Tuple[float, float]] = {}
        max_new = min(
            len(keypoints),
            self.map_max_new_per_frame,
            self.map_max_landmarks,
        )
        for idx in range(max_new):
            kp = keypoints[idx]
            lm_id = self.map_state.new_landmark_id()
            self.map_state.landmarks[lm_id] = MapLandmark(
                id=lm_id,
                position=(float(kp.pt[0]), float(kp.pt[1])),
                descriptor=descriptors[idx].copy(),
                last_seen_frame=self.frame_index,
            )
            landmarks.append(
                Landmark(
                    id=lm_id,
                    x=float(kp.pt[0]),
                    y=float(kp.pt[1]),
                    size=float(kp.size),
                    angle=float(kp.angle),
                    age=1,
                )
            )
            positions[lm_id] = (float(kp.pt[0]), float(kp.pt[1]))
        self.last_map_transform = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        return landmarks, positions

    def _match_map(
        self,
        map_state: MapState,
        keypoints: List[cv2.KeyPoint],
        descriptors: Optional[np.ndarray],
    ) -> Optional[MapMatch]:
        if descriptors is None or len(keypoints) == 0 or not map_state.landmarks:
            return None

        map_ids = list(map_state.landmarks.keys())
        map_descriptors = np.array(
            [map_state.landmarks[lm_id].descriptor for lm_id in map_ids]
        )
        map_points = np.array(
            [map_state.landmarks[lm_id].position for lm_id in map_ids],
            dtype=np.float32,
        )

        matches = self.map_matcher.knnMatch(map_descriptors, descriptors, k=2)
        good_matches: List[cv2.DMatch] = []
        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.map_match_ratio * n.distance:
                good_matches.append(m)

        if len(good_matches) < self.map_min_inliers:
            return None

        src_points = np.array(
            [keypoints[m.trainIdx].pt for m in good_matches], dtype=np.float32
        )
        dst_points = np.array(
            [map_points[m.queryIdx] for m in good_matches], dtype=np.float32
        )
        matrix, inliers = cv2.estimateAffinePartial2D(
            src_points,
            dst_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,
        )
        if matrix is None:
            return None

        if inliers is None:
            inlier_mask = np.ones(len(good_matches), dtype=bool)
        else:
            inlier_mask = inliers.reshape(-1) == 1
        inlier_matches = [
            good_matches[idx]
            for idx, ok in enumerate(inlier_mask)
            if ok
        ]
        if len(inlier_matches) < self.map_min_inliers:
            return None

        return MapMatch(
            transform=matrix,
            inlier_matches=inlier_matches,
            map_ids=map_ids,
        )

    def _update_map_from_match(
        self,
        map_state: MapState,
        match: MapMatch,
        keypoints: List[cv2.KeyPoint],
        descriptors: np.ndarray,
    ) -> Tuple[List[Landmark], Dict[str, Tuple[float, float]]]:
        landmarks: List[Landmark] = []
        positions: Dict[str, Tuple[float, float]] = {}
        matched_frame_indices: Set[int] = set()

        for m in match.inlier_matches:
            map_id = match.map_ids[m.queryIdx]
            kp = keypoints[m.trainIdx]
            map_lm = map_state.landmarks[map_id]
            map_lm.age += 1
            map_lm.last_seen_frame = self.frame_index
            map_lm.descriptor = descriptors[m.trainIdx].copy()
            matched_frame_indices.add(m.trainIdx)

            landmarks.append(
                Landmark(
                    id=map_id,
                    x=float(kp.pt[0]),
                    y=float(kp.pt[1]),
                    size=float(kp.size),
                    angle=float(kp.angle),
                    age=map_lm.age,
                )
            )
            positions[map_id] = (float(kp.pt[0]), float(kp.pt[1]))

        new_count = 0
        if self._should_allow_new_landmarks():
            for idx, kp in enumerate(keypoints):
                if idx in matched_frame_indices:
                    continue
                if len(map_state.landmarks) >= self.map_max_landmarks:
                    break
                if new_count >= self.map_max_new_per_frame:
                    break
                map_pos = self._apply_affine(match.transform, kp.pt)
                lm_id = map_state.new_landmark_id()
                map_state.landmarks[lm_id] = MapLandmark(
                    id=lm_id,
                    position=map_pos,
                    descriptor=descriptors[idx].copy(),
                    last_seen_frame=self.frame_index,
                )
                new_count += 1

                landmarks.append(
                    Landmark(
                        id=lm_id,
                        x=float(kp.pt[0]),
                        y=float(kp.pt[1]),
                        size=float(kp.size),
                        angle=float(kp.angle),
                        age=1,
                    )
                )
                positions[lm_id] = (float(kp.pt[0]), float(kp.pt[1]))

        self.last_map_transform = match.transform
        return landmarks, positions

    def _apply_affine(
        self, matrix: np.ndarray, point: Tuple[float, float]
    ) -> Tuple[float, float]:
        x, y = float(point[0]), float(point[1])
        x_new = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]
        y_new = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]
        return float(x_new), float(y_new)

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
