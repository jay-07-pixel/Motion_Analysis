"""MediaPipe Hands detector initialization and per-frame inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Optional

import numpy as np

from motion_analysis.hands.exceptions import HandsDetectionError, HandsInitializationError
from motion_analysis.hands.landmarks import (
    HandFrame,
    HandsFrame,
    extract_hand_landmarks,
    hand_landmarks_to_pixels,
)
from motion_analysis.hands.model import load_hand_landmarker_model
from motion_analysis.pose.smoothing import KeypointSmoother, SmoothingConfig
from motion_analysis.pose.validation import validate_pixel_keypoints

try:
    import cv2
except ImportError:  # pragma: no cover - OpenCV is a project requirement
    cv2 = None


@dataclass
class HandDetector:
    """Initialized MediaPipe Hand Landmarker plus video timestamps.

    Attributes:
        landmarker: MediaPipe Tasks HandLandmarker in VIDEO running mode.
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


def initialize_hand_detector(
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    num_hands: int = 2,
) -> HandDetector:
    """Create a MediaPipe Hands detector for live video.

    Args:
        min_detection_confidence: Minimum confidence to start a hand
            detection, in [0, 1].
        min_tracking_confidence: Minimum confidence to keep tracking a hand.
        num_hands: Maximum hands to detect. 2 covers left and right.

    Returns:
        A HandDetector wrapping the MediaPipe Hand Landmarker.

    Raises:
        HandsInitializationError: If MediaPipe is missing or the model fails
            to load.
    """
    try:
        import mediapipe as mp
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision import (
            HandLandmarker,
            HandLandmarkerOptions,
            RunningMode,
        )
    except ImportError as exc:
        raise HandsInitializationError(
            "MediaPipe is not installed. Install project requirements with "
            "`pip install -r requirements.txt` and try again."
        ) from exc

    if not hasattr(mp, "Image"):
        raise HandsInitializationError(
            "This MediaPipe installation does not provide the Hand Landmarker "
            "Tasks API. Install mediapipe 0.10.14 or newer."
        )

    model_bytes = load_hand_landmarker_model()
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=model_bytes),
        running_mode=RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    try:
        landmarker = HandLandmarker.create_from_options(options)
    except Exception as exc:
        raise HandsInitializationError(
            f"Failed to initialize MediaPipe Hands: {exc}"
        ) from exc
    return HandDetector(landmarker=landmarker, model_bytes=model_bytes)


def detect_hands(detector: HandDetector, frame_bgr: np.ndarray) -> list[tuple[str, float, Any]]:
    """Run MediaPipe Hands on one BGR frame.

    Args:
        detector: Object returned by initialize_hand_detector().
        frame_bgr: Live RGB frame in OpenCV BGR format.

    Returns:
        A list of ``(handedness, score, landmarks)`` for each detected hand.
        Landmark x/y values are still normalized. The list is empty when no
        hand is found.

    Raises:
        HandsDetectionError: If the frame is invalid or MediaPipe raises.
    """
    if detector is None or getattr(detector, "landmarker", None) is None:
        raise HandsDetectionError("Hand detector is not initialized.")
    if cv2 is None:
        raise HandsDetectionError("OpenCV is required to prepare frames for MediaPipe.")
    if frame_bgr is None or frame_bgr.size == 0:
        raise HandsDetectionError("Cannot run hand detection on an empty frame.")
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise HandsDetectionError(
            f"Expected a BGR color frame with 3 channels, got shape {frame_bgr.shape}."
        )

    try:
        import mediapipe as mp
    except ImportError as exc:
        raise HandsDetectionError(
            "MediaPipe is not installed. Install project requirements and retry."
        ) from exc

    rgb_frame = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    try:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = detector.landmarker.detect_for_video(
            mp_image,
            detector.next_timestamp_ms(),
        )
    except Exception as exc:
        raise HandsDetectionError(
            f"MediaPipe Hands failed while processing a frame: {exc}"
        ) from exc

    if result is None or not result.hand_landmarks:
        return []

    detected: list[tuple[str, float, Any]] = []
    for index, landmarks in enumerate(result.hand_landmarks):
        handedness, score = _read_handedness(result.handedness, index)
        detected.append((handedness, score, landmarks))
    return detected


def close_hand_detector(detector: Optional[HandDetector]) -> None:
    """Release MediaPipe Hands resources.

    Args:
        detector: Detector from initialize_hand_detector(), or None.
    """
    if detector is None:
        return
    landmarker = getattr(detector, "landmarker", detector)
    close = getattr(landmarker, "close", None)
    if callable(close):
        close()
    if isinstance(detector, HandDetector):
        detector.landmarker = None


def _read_handedness(handedness_lists: Any, index: int) -> tuple[str, float]:
    """Read Left/Right label and score for one detected hand.

    Args:
        handedness_lists: ``result.handedness`` from MediaPipe Hands.
        index: Hand index matching ``hand_landmarks``.

    Returns:
        ``(handedness, score)`` with handedness ``Left``, ``Right``, or
        ``Unknown``.
    """
    if handedness_lists is None or index >= len(handedness_lists):
        return "Unknown", 0.0
    categories = handedness_lists[index]
    if not categories:
        return "Unknown", 0.0
    top = categories[0]
    name = (
        getattr(top, "category_name", None)
        or getattr(top, "display_name", None)
        or "Unknown"
    )
    score = float(getattr(top, "score", 0.0) or 0.0)
    if name.lower() == "left":
        return "Left", score
    if name.lower() == "right":
        return "Right", score
    return str(name), score


class HandsEstimator:
    """Owns a MediaPipe Hands detector and returns pixel-space keypoints.

    Separate EMA filters are kept for the left and right hands so a missing
    hand does not corrupt the other hand's history.

    Args:
        min_detection_confidence: Forwarded to initialize_hand_detector().
        min_tracking_confidence: Forwarded to initialize_hand_detector().
        num_hands: Maximum hands to detect. Default 2 for left and right.
        smoothing: EMA and visibility settings. If omitted, SmoothingConfig
            defaults are used.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        num_hands: int = 2,
        smoothing: SmoothingConfig | None = None,
    ) -> None:
        self._detector = initialize_hand_detector(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            num_hands=num_hands,
        )
        self.smoothing = smoothing or SmoothingConfig()
        self._smoothers = {
            "Left": KeypointSmoother(self.smoothing),
            "Right": KeypointSmoother(self.smoothing),
        }

    def estimate(self, frame_bgr: np.ndarray) -> HandsFrame:
        """Detect hands, validate pixels, and return raw plus smoothed keypoints.

        Args:
            frame_bgr: BGR image from the RealSense RGB stream.

        Returns:
            HandsFrame with zero, one, or two hands. Coordinates use the
            current frame size.

        Raises:
            HandsDetectionError: If inference or coordinate conversion fails.
        """
        detections = detect_hands(self._detector, frame_bgr)
        frame_height, frame_width = frame_bgr.shape[:2]
        if not detections:
            for smoother in self._smoothers.values():
                smoother.mark_missing()
            return HandsFrame(
                frame_width=frame_width,
                frame_height=frame_height,
                hands=[],
            )

        seen: set[str] = set()
        hands: list[HandFrame] = []
        for handedness, score, landmarks in detections:
            if handedness in seen:
                continue
            seen.add(handedness)
            hand = self._build_hand(
                handedness,
                score,
                landmarks,
                frame_width,
                frame_height,
            )
            hands.append(hand)

        for side, smoother in self._smoothers.items():
            if side not in seen:
                smoother.mark_missing()

        return HandsFrame(
            frame_width=frame_width,
            frame_height=frame_height,
            hands=hands,
        )

    def reset_tracking(self) -> None:
        """Clear per-hand EMA history without reloading the hands model.

        Call this when an uploaded video restarts so leftover finger
        positions do not appear on the first frames.
        """
        for smoother in self._smoothers.values():
            smoother.reset()

    def close(self) -> None:
        """Close the underlying MediaPipe Hands detector."""
        close_hand_detector(self._detector)
        self._detector = None
        for smoother in self._smoothers.values():
            smoother.reset()

    def _build_hand(
        self,
        handedness: str,
        score: float,
        landmarks: Any,
        frame_width: int,
        frame_height: int,
    ) -> HandFrame:
        """Convert, validate, and smooth one detected hand.

        Args:
            handedness: ``Left`` or ``Right``.
            score: Handedness confidence.
            landmarks: MediaPipe normalized landmarks for this hand.
            frame_width: Current frame width in pixels.
            frame_height: Current frame height in pixels.

        Returns:
            A HandFrame with raw and smoothed pixel keypoints.
        """
        normalized = extract_hand_landmarks(landmarks)
        converted = hand_landmarks_to_pixels(normalized, frame_width, frame_height)
        raw = validate_pixel_keypoints(
            converted,
            frame_width,
            frame_height,
            min_visibility=self.smoothing.min_visibility,
        )
        smoother = self._smoothers.get(handedness)
        if smoother is None:
            smoother = KeypointSmoother(self.smoothing)
            self._smoothers[handedness] = smoother
        smoothed = smoother.update(raw, frame_width, frame_height)
        return HandFrame(
            handedness=handedness,
            score=score,
            frame_width=frame_width,
            frame_height=frame_height,
            normalized_landmarks=normalized,
            raw_keypoints=raw,
            smoothed_keypoints=smoothed,
        )

    def __enter__(self) -> HandsEstimator:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
