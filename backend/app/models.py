from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

"""
Configs for vision detection model & reasoning algorithms
"""
class TransformType(str, Enum):
    similarity = "similarity"
    affine = "affine"
    homography = "homography"

class DetectorType(str, Enum):
    orb = "orb" #fastest
    akaze = "akaze"
    sift = "sift" #most robust

class TransformPriority(str, Enum):
    global_first = "global_first"
    local_first = "local_first"

class LocalTrackingMode(str, Enum):
    patch = "patch"
    annotation_transform = "annotation_transform"
    annotation_points = "annotation_points"
    

"""
Anchors for position tracking
"""
class AnchorKind(str, Enum):
    local = "local"
    global_ = "global"

class GeometryType(str, Enum):
    point = "point"
    polyline = "polyline"
    polygon = "polygon"
    bbox = "bbox"

class Point2D(BaseModel):
    x: float
    y: float

class TransformMatrix(BaseModel):
    matrix: List[List[float]]
    inliers: Optional[int] = None
    reproj_error: Optional[float] = None

class LocalAnchor(BaseModel):
    kind: AnchorKind = AnchorKind.local
    patch_center: Point2D
    patch_radius: int
    keyframe_points: List[Point2D]
    transform_type: TransformType = TransformType.similarity

class GlobalAnchor(BaseModel):
    kind: AnchorKind = AnchorKind.global_
    landmarks: List[LandmarkBinding]
    transform_type: TransformType = TransformType.similarity

class LocalAnchorHint(BaseModel):
    patch_radius: Optional[int] = None
    transform_type: Optional[TransformType] = None

class GlobalAnchorHint(BaseModel):
    k_nearest: Optional[int] = None
    transform_type: Optional[TransformType] = None


"""
Internal reference points used to build world map
"""
class Landmark(BaseModel):
    id: str
    x: float
    y: float
    size: Optional[float] = None
    angle: Optional[float] = None
    age: int = 1

class LandmarkBinding(BaseModel):
    id: str
    keyframe_position: Point2D


"""
Annotations created by client
"""
class AnnotationCreate(BaseModel):
    id: Optional[str] = None
    geometry_type: GeometryType
    points: List[Point2D]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    local_hint: Optional[LocalAnchorHint] = None
    global_hint: Optional[GlobalAnchorHint] = None

class AnnotationRecord(BaseModel):
    id: str
    frame_id: str
    geometry_type: GeometryType
    points: List[Point2D]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    local_anchor: Optional[LocalAnchor] = None
    global_anchor: Optional[GlobalAnchor] = None
    local_transform: Optional[TransformMatrix] = None
    global_transform: Optional[TransformMatrix] = None
    created_at: datetime
    updated_at: datetime

class AnnotationsIn(BaseModel):
    frame_id: Optional[str] = None
    annotations: List[AnnotationCreate]
    timestamp: Optional[datetime] = None


"""
Caching frames to model optical flow
"""
class FramePinRequest(BaseModel):
    frame_id: str

class FrameState(BaseModel):
    frame_id: str
    timestamp: datetime
    landmarks: List[Landmark]
    annotations: List[AnnotationRecord] = Field(default_factory=list)
    
    
"""
Config tracking
"""
class TrackingConfig(BaseModel):
    detector: DetectorType = DetectorType.sift
    transform_priority: TransformPriority = TransformPriority.local_first
    local_tracking_mode: LocalTrackingMode = LocalTrackingMode.annotation_transform
    default_anchor_transform: TransformType = TransformType.similarity
    map_transform: TransformType = TransformType.similarity
    map_match_ratio: float = 0.85
    map_min_inliers: int = 10


class TrackingConfigUpdate(BaseModel):
    detector: Optional[DetectorType] = None
    transform_priority: Optional[TransformPriority] = None
    local_tracking_mode: Optional[LocalTrackingMode] = None
    default_anchor_transform: Optional[TransformType] = None
    map_transform: Optional[TransformType] = None
    map_match_ratio: Optional[float] = None
    map_min_inliers: Optional[int] = None