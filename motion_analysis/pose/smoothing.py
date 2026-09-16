"""Configurable smoothing for valid 2D pixel keypoints.

Raw MediaPipe landmarks jitter from detector noise even when a person is
still. An exponential moving average (EMA) reduces that noise.

Only valid observations update the filter: the landmark must be inside the
current frame and pass the MediaPipe visibility threshold. If a joint
temporarily becomes invalid, the last valid smoothed position is held in
memory and is not drawn, so the skeleton does not jump or invent motion.
"""

from __future__ import annotations

from dataclasses import dataclass

from motion_analysis.pose.exceptions import PoseDetectionError
from motion_analysis.pose.landmarks import PixelKeypoint
from motion_analysis.pose.validation import is_inside_frame, is_visible_enough


@dataclass(frozen=True)
class SmoothingConfig:
    """User-tunable landmark validity and smoothing parameters.

    Pass an instance into PoseEstimator rather than editing filter math
    inside the update function.

    Attributes:
        enabled: If False, smoothed keypoints copy raw values and flags.
        alpha: Weight of the new valid raw sample, in (0, 1]. Smaller values
            reduce jitter more and add more lag.
        min_visibility: Minimum MediaPipe visibility/confidence required for
            a landmark to be valid, drawn, or used to update the filter.
        reset_after_missing_frames: Clear filter history after this many
            consecutive frames with no person detected.
    """

    enabled: bool = True
    alpha: float = 0.35
    min_visibility: float = 0.5
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
    """Per-landmark EMA filter that updates only on valid observations.

    Args:
        config: Validity and smoothing parameters.
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
        """Smooth valid raw keypoints and freeze invalid ones.

        Args:
            raw_keypoints: Validated pixel keypoints from the current frame.
            frame_width: Current frame width in pixels.
            frame_height: Current frame height in pixels.

        Returns:
            Smoothed keypoints in the same order. Invalid landmarks keep
            ``is_valid=False`` so they are not drawn. Raw values are unchanged.

        Raises:
            PoseDetectionError: If smoothing config or frame size is invalid.
        """
        self._missing_frames = 0
        smoothed: list[PixelKeypoint] = []
        for keypoint in raw_keypoints:
            smoothed.append(
                self._smooth_one(keypoint, frame_width, frame_height)
            )
        return smoothed

    def _smooth_one(
        self,
        keypoint: PixelKeypoint,
        frame_width: int,
        frame_height: int,
    ) -> PixelKeypoint:
        """Update one landmark from a valid sample, or hold internal state.

        Args:
            keypoint: Raw validated keypoint for this landmark.
            frame_width: Current frame width in pixels.
            frame_height: Current frame height in pixels.

        Returns:
            Smoothed PixelKeypoint. ``is_valid`` follows the current raw
            observation so a dropped joint is not drawn as a frozen limb.
        """
        previous = self._state.get(keypoint.name)

        if not self.config.enabled:
            return PixelKeypoint(
                name=keypoint.name,
                x=keypoint.x,
                y=keypoint.y,
                visibility=keypoint.visibility,
                in_frame=keypoint.in_frame,
                visible=keypoint.visible,
                is_valid=keypoint.is_valid,
            )

        if keypoint.is_valid:
            if previous is None:
                smoothed_xy = (keypoint.x, keypoint.y)
            else:
                alpha = self.config.alpha
                smoothed_xy = (
                    alpha * keypoint.x + (1.0 - alpha) * previous[0],
                    alpha * keypoint.y + (1.0 - alpha) * previous[1],
                )
            self._state[keypoint.name] = smoothed_xy
            x, y = smoothed_xy
            in_frame = is_inside_frame(x, y, frame_width, frame_height)
            visible = is_visible_enough(keypoint.visibility, self.config.min_visibility)
            return PixelKeypoint(
                name=keypoint.name,
                x=x,
                y=y,
                visibility=keypoint.visibility,
                in_frame=in_frame,
                visible=visible,
                is_valid=in_frame and visible,
            )

        # Invalid this frame: do not update the filter with a bad sample.
        # Keep the last valid position internally, but mark the output invalid
        # so drawing does not show a false frozen joint.
        if previous is not None:
            x, y = previous
        else:
            x, y = keypoint.x, keypoint.y
        return PixelKeypoint(
            name=keypoint.name,
            x=x,
            y=y,
            visibility=keypoint.visibility,
            in_frame=is_inside_frame(x, y, frame_width, frame_height),
            visible=False,
            is_valid=False,
        )
