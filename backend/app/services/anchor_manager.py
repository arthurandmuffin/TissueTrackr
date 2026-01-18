from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional, Set, Tuple
from uuid import uuid4

import cv2
import numpy as np

from ..models import (
    AnnotationCreate,
    AnnotationRecord,
    AnnotationsIn,
    DetectorType,
    FrameState,
    GlobalAnchor,
    GlobalAnchorHint,
    Landmark,
    LandmarkBinding,
    LocalAnchor,
    LocalAnchorHint,
    LocalTrackingMode,
    Point2D,
    TrackingConfig,
    TrackingConfigUpdate,
    TransformMatrix,
    TransformPriority,
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
    reference_gray: Optional[np.ndarray] = None
    map_keyframe_transform: Optional[np.ndarray] = None
    last_tracked_index: Optional[int] = None


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


@dataclass
class FrameSnapshot:
    frame_id: str
    frame_index: int
    gray: np.ndarray
    landmarks: List[Landmark]
    map_transform: Optional[np.ndarray]


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
        frame_cache_size: int = 600,
        detector_type: DetectorType = DetectorType.akaze,
        transform_priority: TransformPriority = TransformPriority.global_first,
        local_tracking_mode: LocalTrackingMode = LocalTrackingMode.patch,
        default_anchor_transform: TransformType = TransformType.similarity,
        map_transform: TransformType = TransformType.similarity,
    ):
        if map_transform == TransformType.homography:
            raise ValueError("Map homography is not supported yet.")
        self.frame_processor = FrameProcessor(
            max_features=max_features,
            max_dimension=max_detection_dimension,
            detector_type=detector_type,
        )
        self.local_patch_radius = local_patch_radius
        self.local_min_points = local_min_points
        self.local_max_points = local_max_points
        self.global_k_nearest = global_k_nearest
        self.min_landmark_age = min_landmark_age
        self.render_landmarks = render_landmarks
        self.transform_priority = transform_priority
        self.local_tracking_mode = local_tracking_mode
        self.default_anchor_transform = default_anchor_transform
        self.map_transform = map_transform
        self.map_max_landmarks = map_max_landmarks
        self.map_max_new_per_frame = map_max_new_per_frame
        self.map_min_inliers = map_min_inliers
        self.map_lost_threshold = map_lost_threshold
        self.map_match_ratio = map_match_ratio
        self.allow_new_landmarks = allow_new_landmarks
        self.freeze_new_landmarks_after_frames = freeze_new_landmarks_after_frames
        self.map_matcher = self._create_matcher(detector_type)
        self.map_state: Optional[MapState] = None
        self.lost_frames = 0
        self.last_map_transform: Optional[np.ndarray] = None
        self.frame_index = 0
        self.last_frame_id: Optional[str] = None
        self.prev_gray: Optional[np.ndarray] = None
        self.last_frame: Optional[np.ndarray] = None
        self.last_landmarks: List[Landmark] = []
        self.annotations: Dict[str, AnnotationState] = {}
        self.frame_cache_size = frame_cache_size
        self.frame_cache_order: Deque[str] = deque()
        self.frame_cache: Dict[str, FrameSnapshot] = {}
        self.pinned_frames: Dict[str, FrameSnapshot] = {}
        self._descriptor_norm = self._norm_for_detector(detector_type)

    def render_frame(
        self,
        frame: np.ndarray,
        landmarks: List[Landmark],
        annotations: Optional[List[AnnotationRecord]] = None,
        frame_id: Optional[str] = None,
    ) -> np.ndarray:
        output_frame = frame.copy()
        total_landmarks = 0
        if self.map_state is not None:
            total_landmarks = len(self.map_state.landmarks)
        # output_frame = self.frame_processor.draw_stats(
        #     output_frame,
        #     self._should_allow_new_landmarks(),
        #     total_landmarks,
        #     frame_id,
        # )
        # if self.render_landmarks:
        #     output_frame = self.frame_processor.draw_landmarks(
        #         output_frame, landmarks
        #     )
        if annotations:
            output_frame = self.frame_processor.draw_annotations(
                output_frame, annotations, transform_priority=self.transform_priority
            )
        return output_frame

    def get_tracking_config(self) -> TrackingConfig:
        return TrackingConfig(
            detector=self.frame_processor.detector_type,
            transform_priority=self.transform_priority,
            local_tracking_mode=self.local_tracking_mode,
            default_anchor_transform=self.default_anchor_transform,
            map_transform=self.map_transform,
            map_match_ratio=self.map_match_ratio,
            map_min_inliers=self.map_min_inliers,
        )

    def update_tracking_config(self, payload: TrackingConfigUpdate) -> TrackingConfig:
        changed = False
        if payload.detector is not None:
            if payload.detector != self.frame_processor.detector_type:
                self.frame_processor.set_detector(payload.detector)
                self._reset_map_state()
                self._reset_matcher_for_detector(payload.detector)
                changed = True
        if payload.transform_priority is not None:
            if payload.transform_priority != self.transform_priority:
                self.transform_priority = payload.transform_priority
                changed = True
        if payload.local_tracking_mode is not None:
            if payload.local_tracking_mode != self.local_tracking_mode:
                self.local_tracking_mode = payload.local_tracking_mode
                changed = True
        if payload.default_anchor_transform is not None:
            if payload.default_anchor_transform != self.default_anchor_transform:
                self.default_anchor_transform = payload.default_anchor_transform
                changed = True
        if payload.map_transform is not None:
            if payload.map_transform == TransformType.homography:
                raise ValueError("Map homography is not supported yet.")
            if payload.map_transform != self.map_transform:
                self.map_transform = payload.map_transform
                changed = True
        if payload.map_match_ratio is not None:
            if payload.map_match_ratio != self.map_match_ratio:
                self.map_match_ratio = payload.map_match_ratio
                changed = True
        if payload.map_min_inliers is not None:
            if payload.map_min_inliers != self.map_min_inliers:
                self.map_min_inliers = payload.map_min_inliers
                changed = True
        if changed:
            self.clear_annotations()
        return self.get_tracking_config()

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
                reference_snapshot = None
                reference_gray = None
                if self.local_tracking_mode != LocalTrackingMode.patch:
                    reference_snapshot = self._get_snapshot(state.record.frame_id)
                    if (
                        reference_snapshot is not None
                        and reference_snapshot.gray.shape != gray.shape
                    ):
                        reference_snapshot = None
                    if reference_snapshot is None and state.reference_gray is not None:
                        if state.reference_gray.shape == gray.shape:
                            reference_gray = state.reference_gray
                if state.local_state is not None:
                    if self.local_tracking_mode == LocalTrackingMode.annotation_points:
                        updated = self._advance_local_points_from_cache(
                            state, gray, update_record_points=True
                        )
                        if not updated:
                            if reference_snapshot is not None:
                                self._update_local_points_from_reference(
                                    state, reference_snapshot.gray, gray
                                )
                            elif reference_gray is not None:
                                self._update_local_points_from_reference(
                                    state, reference_gray, gray
                                )
                            else:
                                self._update_local_points(
                                    state, self.prev_gray, gray
                                )
                        state.record.local_transform = None
                    elif self.local_tracking_mode == LocalTrackingMode.annotation_transform:
                        transform = self._advance_local_transform_from_cache(
                            state, gray
                        )
                        if transform is None:
                            if reference_snapshot is not None:
                                transform = self._update_local_transform_from_reference(
                                    state.local_state,
                                    reference_snapshot.gray,
                                    gray,
                                )
                            elif reference_gray is not None:
                                transform = self._update_local_transform_from_reference(
                                    state.local_state,
                                    reference_gray,
                                    gray,
                                )
                            else:
                                transform = self._update_local_anchor(
                                    state.local_state, self.prev_gray, gray
                                )
                        state.record.local_transform = transform
                    else:
                        state.record.local_transform = self._update_local_anchor(
                            state.local_state, self.prev_gray, gray
                        )
                if state.global_state is not None:
                    if self.local_tracking_mode != LocalTrackingMode.annotation_points:
                        state.record.global_transform = self._update_global_anchor(
                            state.global_state, positions
                        )
                if self.local_tracking_mode == LocalTrackingMode.annotation_points:
                    state.record.global_transform = None
                elif (
                    state.record.local_transform is None
                    and state.record.global_transform is None
                ):
                    state.record.global_transform = self._update_map_anchor(state)
                state.record.updated_at = self._now()

        self.prev_gray = gray
        self.last_frame = frame
        self.last_landmarks = landmarks
        self.last_frame_id = frame_id
        self._cache_snapshot(
            FrameSnapshot(
                frame_id=frame_id,
                frame_index=self.frame_index,
                gray=gray.copy(),
                landmarks=landmarks,
                map_transform=(
                    self.last_map_transform.copy()
                    if self.last_map_transform is not None
                    else None
                ),
            )
        )

        return FrameState(
            frame_id=frame_id,
            timestamp=self._now(),
            landmarks=landmarks,
            annotations=[state.record for state in self.annotations.values()],
        )

    def _cache_snapshot(self, snapshot: FrameSnapshot) -> None:
        if self.frame_cache_size <= 0:
            return
        if snapshot.frame_id in self.pinned_frames:
            return
        if snapshot.frame_id in self.frame_cache:
            self.frame_cache[snapshot.frame_id] = snapshot
            return
        self.frame_cache_order.append(snapshot.frame_id)
        self.frame_cache[snapshot.frame_id] = snapshot
        self._prune_frame_cache()

    def _get_snapshot(self, frame_id: Optional[str]) -> Optional[FrameSnapshot]:
        if frame_id is None:
            return None
        if frame_id in self.pinned_frames:
            return self.pinned_frames[frame_id]
        return self.frame_cache.get(frame_id)

    def _get_snapshot_by_index(self, frame_index: int) -> Optional[FrameSnapshot]:
        for snapshot in self.pinned_frames.values():
            if snapshot.frame_index == frame_index:
                return snapshot
        for snapshot in self.frame_cache.values():
            if snapshot.frame_index == frame_index:
                return snapshot
        return None

    def _get_snapshots_between(
        self, start_index: int, end_index: int
    ) -> List[FrameSnapshot]:
        if start_index >= end_index:
            return []
        snapshots = list(self.pinned_frames.values()) + list(self.frame_cache.values())
        filtered = [
            snap
            for snap in snapshots
            if start_index < snap.frame_index < end_index
        ]
        filtered.sort(key=lambda snap: snap.frame_index)
        return filtered

    def pin_frame(self, frame_id: str) -> None:
        if frame_id in self.pinned_frames:
            return
        snapshot = self.frame_cache.pop(frame_id, None)
        if snapshot is None:
            raise ValueError(f"Frame {frame_id} not in cache.")
        if frame_id in self.frame_cache_order:
            self.frame_cache_order.remove(frame_id)
        self.pinned_frames[frame_id] = snapshot
        self._prune_frame_cache()

    def unpin_frame(self, frame_id: str) -> None:
        self.pinned_frames.pop(frame_id, None)
        self._prune_frame_cache()

    def _prune_frame_cache(self) -> None:
        if self.frame_cache_size <= 0:
            self.frame_cache_order.clear()
            self.frame_cache.clear()
            return
        min_pinned_index = self._min_pinned_index()
        if min_pinned_index is None:
            while len(self.frame_cache_order) > self.frame_cache_size:
                oldest_id = self.frame_cache_order.popleft()
                self.frame_cache.pop(oldest_id, None)
            return
        # Keep all frames from the earliest pin onward; evict only older frames.
        pruned = True
        while pruned:
            pruned = False
            for frame_id in list(self.frame_cache_order):
                snapshot = self.frame_cache.get(frame_id)
                if snapshot is None:
                    self.frame_cache_order.remove(frame_id)
                    pruned = True
                    break
                if snapshot.frame_index < min_pinned_index:
                    self.frame_cache_order.remove(frame_id)
                    self.frame_cache.pop(frame_id, None)
                    pruned = True
                    break

    def _min_pinned_index(self) -> Optional[int]:
        if not self.pinned_frames:
            return None
        return min(snapshot.frame_index for snapshot in self.pinned_frames.values())

    def _reset_map_state(self) -> None:
        self.map_state = None
        self.lost_frames = 0
        self.last_map_transform = None
        self.last_landmarks = []
        self.frame_cache_order.clear()
        self.frame_cache.clear()
        self.pinned_frames.clear()

    def reset_tracking_state(self) -> None:
        self.clear_annotations()
        self._reset_map_state()
        self.prev_gray = None
        self.last_frame = None
        self.last_frame_id = None
        self.frame_index = 0

    def _norm_for_detector(self, detector_type: DetectorType) -> int:
        if detector_type == DetectorType.sift:
            return cv2.NORM_L2
        return cv2.NORM_HAMMING

    def _create_matcher(self, detector_type: DetectorType) -> cv2.BFMatcher:
        return cv2.BFMatcher(self._norm_for_detector(detector_type), crossCheck=False)

    def _reset_matcher_for_detector(self, detector_type: DetectorType) -> None:
        self.map_matcher = self._create_matcher(detector_type)
        self._descriptor_norm = self._norm_for_detector(detector_type)

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
            self.last_map_transform = None
            return [], {}

        if self.map_state is None or not self.map_state.landmarks:
            return self._initialize_map(keypoints, descriptors)

        match = self._match_map(self.map_state, keypoints, descriptors)
        if match is None:
            self.lost_frames = min(self.map_lost_threshold, self.lost_frames + 1)
            self.last_map_transform = None
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
        if self.map_transform == TransformType.affine:
            matrix, inliers = cv2.estimateAffine2D(
                src_points,
                dst_points,
                method=cv2.RANSAC,
                ransacReprojThreshold=5.0,
            )
        else:
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
        snapshot = self._get_snapshot(payload.frame_id)
        if payload.frame_id and snapshot is None:
            raise ValueError("Frame not found in cache or pinned storage.")
        if snapshot is None and self.last_frame is None:
            raise ValueError("No frame available for anchor binding yet.")

        frame_id = payload.frame_id or f"frame-{self.frame_index}"
        is_current_frame = frame_id == self.last_frame_id
        gray_for_local = snapshot.gray if snapshot else None
        if gray_for_local is None and self.last_frame is not None:
            gray_for_local = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY)
        if self.local_tracking_mode == LocalTrackingMode.patch and not is_current_frame:
            gray_for_local = None
        landmarks_for_global = (
            snapshot.landmarks if snapshot is not None else self.last_landmarks
        )
        if snapshot is not None:
            map_keyframe_transform = (
                snapshot.map_transform.copy()
                if snapshot.map_transform is not None
                else None
            )
        else:
            map_keyframe_transform = (
                self.last_map_transform.copy()
                if self.last_map_transform is not None
                else None
            )
        created: List[AnnotationRecord] = []
        for annotation in payload.annotations:
            record = self._build_annotation_record(frame_id, annotation)
            if self.local_tracking_mode == LocalTrackingMode.patch:
                local_anchor, local_state = self._build_local_anchor(
                    record,
                    annotation.local_hint,
                    gray_for_local,
                )
            else:
                local_anchor, local_state = self._build_local_points_anchor(
                    record,
                    annotation.local_hint,
                    gray_for_local,
                )
            global_anchor, global_state = self._build_global_anchor(
                record,
                annotation.global_hint,
                landmarks_for_global,
            )
            record.local_anchor = local_anchor
            record.global_anchor = global_anchor

            tracked_index = None
            if snapshot is not None:
                tracked_index = snapshot.frame_index
            elif self.last_frame_id == frame_id:
                tracked_index = self.frame_index

            self.annotations[record.id] = AnnotationState(
                record=record,
                local_state=local_state,
                global_state=global_state,
                reference_gray=gray_for_local.copy()
                if gray_for_local is not None
                else None,
                map_keyframe_transform=map_keyframe_transform,
                last_tracked_index=tracked_index,
            )
            created.append(record)
        return created

    def clear_annotations(self) -> int:
        count = len(self.annotations)
        self.annotations.clear()
        return count

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
        self,
        record: AnnotationRecord,
        hint: Optional[LocalAnchorHint],
        gray_frame: Optional[np.ndarray],
    ) -> Tuple[Optional[LocalAnchor], Optional[LocalAnchorState]]:
        if gray_frame is None or not record.points:
            return None, None

        transform_type = (
            hint.transform_type
            if hint and hint.transform_type
            else self.default_anchor_transform
        )
        radius = hint.patch_radius if hint and hint.patch_radius else self.local_patch_radius
        radius = int(max(64, min(128, radius)))

        center = self._centroid(record.points)
        x0, y0, x1, y1 = self._patch_bounds(center, radius, gray_frame.shape)
        roi = gray_frame[y0:y1, x0:x1]

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

    def _build_local_points_anchor(
        self,
        record: AnnotationRecord,
        hint: Optional[LocalAnchorHint],
        gray_frame: Optional[np.ndarray],
    ) -> Tuple[Optional[LocalAnchor], Optional[LocalAnchorState]]:
        if gray_frame is None or not record.points:
            return None, None

        transform_type = (
            hint.transform_type
            if hint and hint.transform_type
            else self.default_anchor_transform
        )
        center = self._centroid(record.points)
        points = np.array(
            [[pt.x, pt.y] for pt in record.points], dtype=np.float32
        ).reshape(-1, 1, 2)

        local_anchor = LocalAnchor(
            patch_center=Point2D(x=center[0], y=center[1]),
            patch_radius=0,
            keyframe_points=self._points_to_model(points),
            transform_type=transform_type,
        )
        state = LocalAnchorState(
            keyframe_points=points.copy(),
            last_points=points.copy(),
            transform_type=transform_type,
            patch_center=center,
            patch_radius=0,
        )
        return local_anchor, state

    def _build_global_anchor(
        self,
        record: AnnotationRecord,
        hint: Optional[GlobalAnchorHint],
        landmarks: List[Landmark],
    ) -> Tuple[Optional[GlobalAnchor], Optional[GlobalAnchorState]]:
        if not landmarks or not record.points:
            return None, None

        transform_type = (
            hint.transform_type
            if hint and hint.transform_type
            else self.default_anchor_transform
        )
        k_nearest = hint.k_nearest if hint and hint.k_nearest else self.global_k_nearest
        k_nearest = max(6, min(20, int(k_nearest)))

        center = self._centroid(record.points)
        candidates = [lm for lm in landmarks if lm.age >= self.min_landmark_age]
        if not candidates:
            candidates = landmarks

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
            state.last_points = new_points[mask]
            state.keyframe_points = state.keyframe_points[mask]
            if mask.sum() == 0:
                return None
            return self._translation_transform(
                state.keyframe_points, state.last_points
            )

        state.last_points = new_points[mask]
        state.keyframe_points = state.keyframe_points[mask]

        return self._estimate_transform(
            state.keyframe_points.reshape(-1, 2),
            state.last_points.reshape(-1, 2),
            state.transform_type,
        )

    def _update_local_points(
        self,
        state: AnnotationState,
        prev_gray: np.ndarray,
        gray: np.ndarray,
    ) -> None:
        local_state = state.local_state
        if local_state is None or local_state.last_points.size == 0:
            return

        new_points, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            gray,
            local_state.last_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
        )
        if new_points is None or status is None:
            return

        mask = status.reshape(-1) == 1
        updated = local_state.last_points.copy()
        updated[mask] = new_points[mask]
        local_state.last_points = updated

        state.record.points = self._points_to_model(updated)

    def _update_local_points_from_reference(
        self,
        state: AnnotationState,
        reference_gray: np.ndarray,
        gray: np.ndarray,
    ) -> None:
        local_state = state.local_state
        if local_state is None or local_state.keyframe_points.size == 0:
            return

        new_points, status, _ = cv2.calcOpticalFlowPyrLK(
            reference_gray,
            gray,
            local_state.keyframe_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
        )
        if new_points is None or status is None:
            return

        mask = status.reshape(-1) == 1
        updated = local_state.last_points.copy()
        if updated.shape != new_points.shape:
            updated = new_points.copy()
        updated[mask] = new_points[mask]
        local_state.last_points = updated
        state.record.points = self._points_to_model(updated)

    def _track_points(
        self, prev_gray: np.ndarray, gray: np.ndarray, points: np.ndarray
    ) -> np.ndarray:
        new_points, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            gray,
            points,
            None,
            winSize=(21, 21),
            maxLevel=3,
        )
        if new_points is None or status is None:
            return points
        mask = status.reshape(-1) == 1
        updated = points.copy()
        updated[mask] = new_points[mask]
        return updated

    def _advance_local_points_from_cache(
        self,
        state: AnnotationState,
        current_gray: np.ndarray,
        update_record_points: bool,
    ) -> bool:
        local_state = state.local_state
        if local_state is None or local_state.last_points.size == 0:
            return False

        start_index = state.last_tracked_index
        start_gray = None
        points = local_state.last_points
        if start_index is not None:
            snapshot = self._get_snapshot_by_index(start_index)
            if snapshot is not None:
                start_gray = snapshot.gray
            elif state.reference_gray is not None:
                if state.reference_gray.shape == current_gray.shape:
                    start_gray = state.reference_gray
                    start_index = None
                    points = local_state.keyframe_points
        if start_gray is None and state.reference_gray is not None:
            if state.reference_gray.shape == current_gray.shape:
                start_gray = state.reference_gray
                start_index = None
                points = local_state.keyframe_points

        if start_gray is None:
            if self.prev_gray is None:
                return False
            start_gray = self.prev_gray
            start_index = self.frame_index - 1

        if start_index is None:
            points = self._track_points(start_gray, current_gray, points)
            local_state.last_points = points
            state.last_tracked_index = self.frame_index
            if update_record_points:
                state.record.points = self._points_to_model(points)
            return True
        for snapshot in self._get_snapshots_between(start_index, self.frame_index):
            points = self._track_points(start_gray, snapshot.gray, points)
            start_gray = snapshot.gray
            start_index = snapshot.frame_index

        if start_index != self.frame_index:
            points = self._track_points(start_gray, current_gray, points)
            start_index = self.frame_index

        local_state.last_points = points
        state.last_tracked_index = start_index
        if update_record_points:
            state.record.points = self._points_to_model(points)
        return True

    def _advance_local_transform_from_cache(
        self,
        state: AnnotationState,
        current_gray: np.ndarray,
    ) -> Optional[TransformMatrix]:
        if not self._advance_local_points_from_cache(
            state, current_gray, update_record_points=False
        ):
            return None
        local_state = state.local_state
        if local_state is None or local_state.keyframe_points.size == 0:
            return None
        if local_state.last_points.shape[0] < 4:
            return self._translation_transform(
                local_state.keyframe_points, local_state.last_points
            )
        return self._estimate_transform(
            local_state.keyframe_points.reshape(-1, 2),
            local_state.last_points.reshape(-1, 2),
            local_state.transform_type,
        )

    def _update_local_transform_from_reference(
        self,
        local_state: LocalAnchorState,
        reference_gray: np.ndarray,
        gray: np.ndarray,
    ) -> Optional[TransformMatrix]:
        if local_state.keyframe_points.size == 0:
            return None

        new_points, status, _ = cv2.calcOpticalFlowPyrLK(
            reference_gray,
            gray,
            local_state.keyframe_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
        )
        if new_points is None or status is None:
            return None

        mask = status.reshape(-1) == 1
        if mask.sum() < 4:
            local_state.last_points = new_points[mask]
            local_state.keyframe_points = local_state.keyframe_points[mask]
            if mask.sum() == 0:
                return None
            return self._translation_transform(
                local_state.keyframe_points, local_state.last_points
            )

        local_state.last_points = new_points[mask]
        local_state.keyframe_points = local_state.keyframe_points[mask]

        return self._estimate_transform(
            local_state.keyframe_points.reshape(-1, 2),
            local_state.last_points.reshape(-1, 2),
            local_state.transform_type,
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
            if len(src_points) == 0:
                return None
            return self._translation_transform(
                np.array(src_points, dtype=np.float32),
                np.array(dst_points, dtype=np.float32),
            )

        return self._estimate_transform(
            np.array(src_points, dtype=np.float32),
            np.array(dst_points, dtype=np.float32),
            state.transform_type,
        )

    def _update_map_anchor(
        self, state: AnnotationState
    ) -> Optional[TransformMatrix]:
        if state.map_keyframe_transform is None or self.last_map_transform is None:
            return None
        inv_current = self._invert_affine(self.last_map_transform)
        if inv_current is None:
            return None
        composed = self._compose_affine(inv_current, state.map_keyframe_transform)
        if composed is None:
            return None
        return TransformMatrix(matrix=composed.tolist())

    def _invert_affine(self, matrix: np.ndarray) -> Optional[np.ndarray]:
        if matrix.shape != (2, 3):
            return None
        affine = np.vstack([matrix, [0.0, 0.0, 1.0]])
        try:
            inv = np.linalg.inv(affine)
        except np.linalg.LinAlgError:
            return None
        return inv[:2, :]

    def _compose_affine(
        self, left: np.ndarray, right: np.ndarray
    ) -> Optional[np.ndarray]:
        if left.shape != (2, 3) or right.shape != (2, 3):
            return None
        left_3 = np.vstack([left, [0.0, 0.0, 1.0]])
        right_3 = np.vstack([right, [0.0, 0.0, 1.0]])
        composed = left_3 @ right_3
        return composed[:2, :]

    def _translation_transform(
        self, src_points: np.ndarray, dst_points: np.ndarray
    ) -> Optional[TransformMatrix]:
        src = src_points.reshape(-1, 2)
        dst = dst_points.reshape(-1, 2)
        if src.size == 0 or dst.size == 0:
            return None
        delta = dst - src
        dx = float(delta[:, 0].mean())
        dy = float(delta[:, 1].mean())
        return TransformMatrix(
            matrix=[[1.0, 0.0, dx], [0.0, 1.0, dy]],
            inliers=int(src.shape[0]),
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
