"""MediaPipe Pose detector initialization and per-frame inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Optional

import numpy as np

from motion_analysis.pose.exceptions import PoseDetectionError, PoseInitializationError
from motion_analysis.pose.landmarks import (
    PoseFrame,
    convert_to_pixel_coordinates,
    extract_landmarks,
)
from motion_analysis.pose.model import load_pose_landmarker_model

try:
    import cv2
except ImportError:  # pragma: no cover - OpenCV is a project requirement
    cv2 = None


@dataclass
class PoseDetector:
    """Initialized MediaPipe Pose Landmarker plus video timestamps.

    Attributes:
        landmarker: MediaPipe Tasks PoseLandmarker in VIDEO running mode.
        model_bytes: Model buffer kept alive for the C API.
        origin_time: perf_counter value used to create monotonic timestamps.
        last_timestamp_ms: Last timestamp passed to detect_for_video().
    """

    landmarker: Any
    model_bytes: bytes
    origin_time: float = field(default_factory=perf_counter)
    last_timestamp_ms: int = -1

    def next_timestamp_ms(self) -> int:
        """Return a strictly increasing millisecond timestamp.

        Returns:
            Timestamp suitable for MediaPipe VIDEO mode.
        """
        timestamp_ms = int((perf_counter() - self.origin_time) * 1000)
        if timestamp_ms <= self.last_timestamp_ms:
            timestamp_ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = timestamp_ms
        return timestamp_ms


def initialize_pose_detector(
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    model_complexity: int = 1,
) -> PoseDetector:
    """Create a MediaPipe Pose detector for live video.

    Args:
        min_detection_confidence: Minimum confidence to accept a person
            detection, in [0, 1].
        min_tracking_confidence: Minimum confidence to keep tracking the
            current person, in [0, 1].
        model_complexity: Unused by the current Tasks API. Kept so later
            heavy/full model selection can be added without changing callers.

    Returns:
        A PoseDetector wrapping the MediaPipe Pose Landmarker. Close it with
        close_pose_detector() when finished.

    Raises:
        PoseInitializationError: If MediaPipe is missing or the model fails
            to load.
    """
    del model_complexity  # Lite landmarker is used for live RGB capture.
    try:
        import mediapipe as mp
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision import (
            PoseLandmarker,
            PoseLandmarkerOptions,
            RunningMode,
        )
    except ImportError as exc:
        raise PoseInitializationError(
            "MediaPipe is not installed. Install project requirements with "
            "`pip install -r requirements.txt` and try again."
        ) from exc

    if not hasattr(mp, "Image"):
        raise PoseInitializationError(
            "This MediaPipe installation does not provide the Pose Landmarker "
            "Tasks API. Install mediapipe 0.10.14 or newer."
        )

    model_bytes = load_pose_landmarker_model()
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=model_bytes),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=min_detection_confidence,
        min_pose_presence_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_segmentation_masks=False,
    )
    try:
        landmarker = PoseLandmarker.create_from_options(options)
    except Exception as exc:
        raise PoseInitializationError(
            f"Failed to initialize MediaPipe Pose: {exc}"
        ) from exc
    return PoseDetector(landmarker=landmarker, model_bytes=model_bytes)


def detect_pose(detector: PoseDetector, frame_bgr: np.ndarray) -> Optional[Any]:
    """Run MediaPipe Pose on one BGR frame.

    Args:
        detector: Object returned by initialize_pose_detector().
        frame_bgr: Live RGB frame in OpenCV BGR format.

    Returns:
        The main person's normalized landmark list when a person is found,
        otherwise None. Landmark x/y values are still normalized 0-1.

    Raises:
        PoseDetectionError: If the frame is invalid or MediaPipe raises.
    """
    if detector is None or getattr(detector, "landmarker", None) is None:
        raise PoseDetectionError("Pose detector is not initialized.")
    if cv2 is None:
        raise PoseDetectionError("OpenCV is required to prepare frames for MediaPipe.")
    if frame_bgr is None or frame_bgr.size == 0:
        raise PoseDetectionError("Cannot run pose detection on an empty frame.")
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise PoseDetectionError(
            f"Expected a BGR color frame with 3 channels, got shape {frame_bgr.shape}."
        )

    try:
        import mediapipe as mp
    except ImportError as exc:
        raise PoseDetectionError(
            "MediaPipe is not installed. Install project requirements and retry."
        ) from exc

    # MediaPipe Pose expects RGB. The camera pipeline delivers BGR for OpenCV.
    rgb_frame = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    try:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = detector.landmarker.detect_for_video(
            mp_image,
            detector.next_timestamp_ms(),
        )
    except Exception as exc:
        raise PoseDetectionError(
            f"MediaPipe Pose failed while processing a frame: {exc}"
        ) from exc

    if result is None or not result.pose_landmarks:
        return None
    # Only the primary person is used in this step; no 3D/world landmarks.
    return result.pose_landmarks[0]


def close_pose_detector(detector: Optional[PoseDetector]) -> None:
    """Release MediaPipe Pose resources.

    Args:
        detector: Detector from initialize_pose_detector(), or None.
    """
    if detector is None:
        return
    landmarker = getattr(detector, "landmarker", detector)
    close = getattr(landmarker, "close", None)
    if callable(close):
        close()
    if isinstance(detector, PoseDetector):
        detector.landmarker = None


class PoseEstimator:
    """Owns a MediaPipe Pose detector and returns pixel-space keypoints.

    Args:
        min_detection_confidence: Forwarded to initialize_pose_detector().
        min_tracking_confidence: Forwarded to initialize_pose_detector().
        model_complexity: Forwarded to initialize_pose_detector().
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
    ) -> None:
        self._detector = initialize_pose_detector(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity,
        )

    def estimate(self, frame_bgr: np.ndarray) -> Optional[PoseFrame]:
        """Detect pose and convert landmarks to pixel coordinates.

        Width and height are taken from the current frame shape so coordinate
        conversion follows the live RealSense resolution.

        Args:
            frame_bgr: BGR image from the RealSense RGB stream.

        Returns:
            PoseFrame with normalized landmarks and pixel keypoints, or None
            when no person is detected.

        Raises:
            PoseDetectionError: If inference or coordinate conversion fails.
        """
        pose_landmarks = detect_pose(self._detector, frame_bgr)
        if pose_landmarks is None:
            return None

        # Use the live frame size, never a hard-coded resolution.
        frame_height, frame_width = frame_bgr.shape[:2]
        normalized = extract_landmarks(pose_landmarks)
        pixels = convert_to_pixel_coordinates(normalized, frame_width, frame_height)
        return PoseFrame(
            frame_width=frame_width,
            frame_height=frame_height,
            normalized_landmarks=normalized,
            pixel_keypoints=pixels,
        )

    def close(self) -> None:
        """Close the underlying MediaPipe Pose detector."""
        close_pose_detector(self._detector)
        self._detector = None

    def __enter__(self) -> PoseEstimator:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
