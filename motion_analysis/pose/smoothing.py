"""Configurable smoothing for 2D pixel keypoints.

Raw MediaPipe landmarks jitter from detector noise, lighting, and small
pose-model updates even when the person is standing still. An exponential
moving average (EMA) reduces that frame-to-frame noise while keeping a
separate copy of the raw coordinates.

Out-of-frame raw samples are not clamped. By default they also do not update
the smoother, so an inferred foot below the image does not pull the filtered
keypoint off-screen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from motion_analysis.pose.exceptions import PoseDetectionError
from motion_analysis.pose.landmarks import PixelKeypoint
from motion_analysis.pose.validation import is_inside_frame


@dataclass(frozen=True)
class SmoothingConfig:
    """User-tunable 2D keypoint smoothing parameters.

    Pass an instance into PoseEstimator rather than editing filter math
    inside the update function.

    Attributes:
        enabled: If False, smoothed keypoints are copies of the raw values.
        alpha: Weight of the new raw sample, in (0, 1]. Smaller values
            reduce jitter more and add more lag. 0.35 is a live-camera default.
        min_visibility: Raw samples below this visibility do not update the
            filter. The previous smoothed position is held instead.
        hold_when_out_of_frame: If True, out-of-frame raw samples do not
            update the filter. The last in-frame smoothed position is held.
        reset_after_missing_frames: Clear filter history after this many
            consecutive frames with no person detected.
    """

    enabled: bool = True
    alpha: float = 0.35
    min_visibility: float = 0.5
    hold_when_out_of_frame: bool = True
    reset_after_missing_frames: int = 8

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha <= 1.0):
            raise PoseDetectionError(
                f"Smoothing alpha must be in (0, 1], got {self.alpha}."
            )
        if not (0.0 <= self.min_visibility <= 1.0):
            raise PoseDetectionError(
                "Smoothing min_visibility must be in [0, 1], "
                f"got {self.min_visibility}."
            )
        if self.reset_after_missing_frames < 1:
            raise PoseDetectionError(
                "reset_after_missing_frames must be >= 1, "
                f"got {self.reset_after_missing_frames}."
            )


class KeypointSmoother:
    """Per-landmark EMA filter that keeps raw and smoothed coordinates.

    Args:
        config: Smoothing parameters. Stored on the instance so callers can
            change behavior without editing this class.
    """

    def __init__(self, config: SmoothingConfig | None = None) -> None:
        self.config = config or SmoothingConfig()
        self._state: dict[str, tuple[float, float]] = {}
        self._missing_frames = 0

    def mark_missing(self) -> None:
        """Record that this frame had no person.

        After ``reset_after_missing_frames`` misses, history is cleared so a
        new person does not inherit the previous pose.
        """
        self._missing_frames += 1
        if self._missing_frames >= self.config.reset_after_missing_frames:
            self.reset()

    def reset(self) -> None:
        """Forget all per-landmark smoothing history."""
        self._state.clear()
        self._missing_frames = 0

    def update(
        self,
        raw_keypoints: list[PixelKeypoint],
        frame_width: int,
        frame_height: int,
    ) -> list[PixelKeypoint]:
        """Smooth one frame of raw pixel keypoints.

        Args:
            raw_keypoints: Validated pixel keypoints from the current frame.
            frame_width: Current frame width in pixels.
            frame_height: Current frame height in pixels.

        Returns:
            Smoothed keypoints in the same landmark order. Raw values are not
            modified.

        Raises:
            PoseDetectionError: If smoothing config or frame size is invalid.
        """
        if not self.config.enabled:
            return [
                PixelKeypoint(
                    name=keypoint.name,
                    x=keypoint.x,
                    y=keypoint.y,
                    visibility=keypoint.visibility,
                    in_frame=keypoint.in_frame,
                )
                for keypoint in raw_keypoints
            ]

        self._missing_frames = 0
        smoothed: list[PixelKeypoint] = []
        for keypoint in raw_keypoints:
            x, y = self._smooth_one(keypoint)
            smoothed.append(
                PixelKeypoint(
                    name=keypoint.name,
                    x=x,
                    y=y,
                    visibility=keypoint.visibility,
                    in_frame=is_inside_frame(x, y, frame_width, frame_height),
                )
            )
        return smoothed

    def _smooth_one(self, keypoint: PixelKeypoint) -> tuple[float, float]:
        """Apply EMA to one landmark, or hold the previous value.

        Args:
            keypoint: Raw validated keypoint for this landmark.

        Returns:
            Smoothed ``(x, y)`` in pixels.
        """
        previous = self._state.get(keypoint.name)
        if self._should_accept_raw(keypoint):
            if previous is None:
                smoothed = (keypoint.x, keypoint.y)
            else:
                alpha = self.config.alpha
                smoothed = (
                    alpha * keypoint.x + (1.0 - alpha) * previous[0],
                    alpha * keypoint.y + (1.0 - alpha) * previous[1],
                )
            self._state[keypoint.name] = smoothed
            return smoothed

        if previous is not None:
            return previous
        # No history yet: report the raw estimate but do not seed the filter
        # with an out-of-frame or low-visibility point.
        return keypoint.x, keypoint.y

    def _should_accept_raw(self, keypoint: PixelKeypoint) -> bool:
        """Return whether this raw sample may update the filter.

        Args:
            keypoint: Raw validated keypoint.

        Returns:
            True if the sample is finite, visible enough, and optionally
            inside the frame.
        """
        if not math.isfinite(keypoint.x) or not math.isfinite(keypoint.y):
            return False
        if keypoint.visibility < self.config.min_visibility:
            return False
        if self.config.hold_when_out_of_frame and not keypoint.in_frame:
            return False
        return True
