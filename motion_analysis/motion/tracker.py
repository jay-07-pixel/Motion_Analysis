"""Per-landmark 2D motion tracking from smoothed pixel keypoints.

Every valid tracked keypoint gets position, displacement, and distance
travelled. Invalid or missing landmarks skip movement math for that frame
so a dropped joint does not invent a jump.

Live camera and uploaded video both feed this tracker after the shared
RGB → pose/hands → validation → smoothing pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from motion_analysis.motion.geometry import (
    Point2D,
    accumulate_distance,
    displacement_magnitude,
    displacement_vector,
    position,
)
from motion_analysis.motion.landmarks import (
    KeypointObservation,
    LandmarkTarget,
    collect_observations,
    default_landmark,
    selectable_landmarks,
)
from motion_analysis.hands.landmarks import HandsFrame
from motion_analysis.pose.landmarks import PoseFrame
from motion_analysis.pose.smoothing import SmoothingConfig
from motion_analysis.pose.validation import describe_invalid_reason


@dataclass
class _LandmarkState:
    """Internal running totals for one landmark.

    Attributes:
        initial: First valid smoothed position (P_initial).
        previous: Smoothed position from the immediately previous *processed*
            frame when that frame was valid; otherwise None for gap handling.
        previous_frame_index: Processor frame index of the last update.
        previous_was_valid: True if that previous processed frame was valid.
        distance_travelled: Sum of consecutive valid-frame segments.
        last_valid_position: Most recent valid smoothed position.
        last_valid_raw: Most recent valid raw position, for comparison.
        last_valid_timestamp: Source timestamp of the last valid sample.
    """

    initial: Optional[Point2D] = None
    previous: Optional[Point2D] = None
    previous_frame_index: Optional[int] = None
    previous_was_valid: bool = False
    distance_travelled: float = 0.0
    last_valid_position: Optional[Point2D] = None
    last_valid_raw: Optional[Point2D] = None
    last_valid_timestamp: Optional[float] = None


@dataclass(frozen=True)
class MotionMeasurement:
    """Motion snapshot for the landmark currently shown on screen.

    Attributes:
        landmark: User-selected body or hand target.
        is_valid: True when this frame's smoothed keypoint may be used.
        invalid_reason: Why movement was skipped, if invalid.
        timestamp_seconds: Source timestamp for this processed frame.
        source_fps: Actual FPS reported by the live camera or video file.
        position: Current smoothed ``P = (x, y)``, or None if invalid.
        raw_position: Current raw ``(x, y)`` when the raw sample is valid.
        initial_position: First valid smoothed position, if one exists.
        displacement: ``ΔP = P_current - P_initial`` when current is valid.
        displacement_magnitude: ``|ΔP|`` in pixels when current is valid.
        distance_travelled: Path length accumulated so far, in pixels.
        sample_count: Number of valid samples included in the path.
    """

    landmark: LandmarkTarget
    is_valid: bool
    invalid_reason: Optional[str]
    timestamp_seconds: float
    source_fps: float
    position: Optional[Point2D]
    raw_position: Optional[Point2D]
    initial_position: Optional[Point2D]
    displacement: Optional[Point2D]
    displacement_magnitude: Optional[float]
    distance_travelled: float
    sample_count: int


class MotionTracker:
    """Tracks 2D position, displacement, and path length for all landmarks.

    Args:
        min_visibility: Copied into invalid-reason text; motion math still
            trusts ``PixelKeypoint.is_valid`` from the shared validator.
    """

    def __init__(self, min_visibility: float = 0.5) -> None:
        self._min_visibility = min_visibility
        self._states: dict[str, _LandmarkState] = {}
        self._sample_counts: dict[str, int] = {}
        self._last_timestamp = 0.0
        self._last_source_fps = 0.0
        self._last_reasons: dict[str, Optional[str]] = {}
        self._last_current_raw: dict[str, Optional[Point2D]] = {}
        self._last_current_smoothed: dict[str, Optional[Point2D]] = {}
        self._last_valid_flags: dict[str, bool] = {}

    def reset(self) -> None:
        """Clear every landmark's initial position and path length."""
        self._states.clear()
        self._sample_counts.clear()
        self._last_reasons.clear()
        self._last_current_raw.clear()
        self._last_current_smoothed.clear()
        self._last_valid_flags.clear()

    def update(
        self,
        pose_frame: Optional[PoseFrame],
        hands_frame: Optional[HandsFrame],
        timestamp_seconds: float,
        source_fps: float,
        frame_index: int,
        smoothing: Optional[SmoothingConfig] = None,
    ) -> None:
        """Update every catalog landmark from one processed RGB frame.

        Args:
            pose_frame: Shared pipeline body result, or None.
            hands_frame: Shared pipeline hands result, or None.
            timestamp_seconds: Actual frame time from the camera or file.
                Not a hard-coded clock rate.
            source_fps: FPS reported by the active source (0 if unknown).
            frame_index: Monotonic index of this ``process()`` call.
            smoothing: Optional config used only for invalid-reason text.
        """
        min_visibility = (
            smoothing.min_visibility if smoothing is not None else self._min_visibility
        )
        self._last_timestamp = timestamp_seconds
        self._last_source_fps = source_fps
        observations = collect_observations(pose_frame, hands_frame)

        for target in selectable_landmarks():
            observation = observations.get(target.key)
            self._update_one(
                target.key,
                observation,
                timestamp_seconds,
                frame_index,
                min_visibility,
            )

    def snapshot(self, landmark: LandmarkTarget | None = None) -> MotionMeasurement:
        """Return the displayed measurement for one landmark.

        Args:
            landmark: Landmark to report. Defaults to LEFT_WRIST.

        Returns:
            Values computed from smoothed pixels. Raw position is included
            for comparison and is not used in displacement or distance.
        """
        target = landmark or default_landmark()
        state = self._states.get(target.key, _LandmarkState())
        is_valid = self._last_valid_flags.get(target.key, False)
        current = self._last_current_smoothed.get(target.key)
        raw = self._last_current_raw.get(target.key)
        reason = self._last_reasons.get(target.key)

        displacement = None
        magnitude = None
        if is_valid and current is not None and state.initial is not None:
            displacement = displacement_vector(current, state.initial)
            magnitude = displacement_magnitude(current, state.initial)

        return MotionMeasurement(
            landmark=target,
            is_valid=is_valid,
            invalid_reason=None if is_valid else (reason or "not detected"),
            timestamp_seconds=self._last_timestamp,
            source_fps=self._last_source_fps,
            position=current if is_valid else None,
            raw_position=raw if is_valid else None,
            initial_position=state.initial,
            displacement=displacement,
            displacement_magnitude=magnitude,
            distance_travelled=state.distance_travelled,
            sample_count=self._sample_counts.get(target.key, 0),
        )

    def _update_one(
        self,
        key: str,
        observation: Optional[KeypointObservation],
        timestamp_seconds: float,
        frame_index: int,
        min_visibility: float,
    ) -> None:
        """Apply one landmark's observation, or skip movement if invalid.

        Args:
            key: Catalog key for this landmark.
            observation: Raw/smoothed pair, or None if the joint is absent.
            timestamp_seconds: Source timestamp of this frame.
            frame_index: Processor frame index of this update.
            min_visibility: Threshold used in the invalid-reason string.
        """
        state = self._states.setdefault(key, _LandmarkState())
        smoothed = observation.smoothed if observation is not None else None
        raw = observation.raw if observation is not None else None
        is_valid = smoothed is not None and smoothed.is_valid

        if not is_valid:
            reason = "not detected"
            if smoothed is not None:
                reason = describe_invalid_reason(smoothed, min_visibility)
            elif raw is not None:
                reason = describe_invalid_reason(raw, min_visibility)
            self._last_reasons[key] = reason
            self._last_valid_flags[key] = False
            self._last_current_smoothed[key] = None
            self._last_current_raw[key] = None
            state.previous = None
            state.previous_was_valid = False
            state.previous_frame_index = frame_index
            return

        current = position(smoothed.x, smoothed.y)
        raw_point = None
        if raw is not None and raw.is_valid:
            raw_point = position(raw.x, raw.y)

        if state.initial is None:
            state.initial = current

        consecutive = (
            state.previous_was_valid
            and state.previous is not None
            and state.previous_frame_index is not None
            and frame_index == state.previous_frame_index + 1
        )
        if consecutive:
            state.distance_travelled = accumulate_distance(
                state.distance_travelled,
                state.previous,
                current,
            )

        state.previous = current
        state.previous_was_valid = True
        state.previous_frame_index = frame_index
        state.last_valid_position = current
        state.last_valid_raw = raw_point
        state.last_valid_timestamp = timestamp_seconds
        self._sample_counts[key] = self._sample_counts.get(key, 0) + 1
        self._last_reasons[key] = None
        self._last_valid_flags[key] = True
        self._last_current_smoothed[key] = current
        self._last_current_raw[key] = raw_point
