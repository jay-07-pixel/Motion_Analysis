"""Validate 2D pixel keypoints against the live frame and MediaPipe visibility.

A landmark is valid only when both are true:

- its pixel coordinates lie inside the current frame
- MediaPipe visibility/confidence is at or above a configurable threshold

Out-of-frame estimates are kept internally and are not clamped onto the
image border. Invalid landmarks are not drawn and do not update smoothing.
"""

from __future__ import annotations

import math

from motion_analysis.pose.exceptions import PoseDetectionError
from motion_analysis.pose.landmarks import PixelKeypoint


def is_inside_frame(
    x: float,
    y: float,
    frame_width: int,
    frame_height: int,
) -> bool:
    """Return whether a pixel coordinate falls on a visible pixel.

    Visible pixels occupy the half-open rectangle
    ``[0, frame_width) x [0, frame_height)``. A value equal to the width or
    height is already outside the last valid pixel index.

    Args:
        x: Horizontal pixel coordinate.
        y: Vertical pixel coordinate.
        frame_width: Current frame width in pixels.
        frame_height: Current frame height in pixels.

    Returns:
        True if ``(x, y)`` is a finite point inside the visible image.

    Raises:
        PoseDetectionError: If the frame size is invalid.
    """
    _require_frame_size(frame_width, frame_height)
    if not math.isfinite(x) or not math.isfinite(y):
        return False
    return 0.0 <= x < frame_width and 0.0 <= y < frame_height


def is_visible_enough(visibility: float, min_visibility: float) -> bool:
    """Return whether MediaPipe visibility passes the configured threshold.

    Args:
        visibility: MediaPipe visibility/confidence in [0, 1].
        min_visibility: Minimum score that counts as visible.

    Returns:
        True if ``visibility`` is finite and at least ``min_visibility``.
    """
    if not math.isfinite(visibility):
        return False
    return visibility >= min_visibility


def is_landmark_valid(
    x: float,
    y: float,
    visibility: float,
    frame_width: int,
    frame_height: int,
    min_visibility: float,
) -> bool:
    """Return whether a landmark may be drawn, displayed, or smoothed.

    Args:
        x: Horizontal pixel coordinate.
        y: Vertical pixel coordinate.
        visibility: MediaPipe visibility/confidence.
        frame_width: Current frame width in pixels.
        frame_height: Current frame height in pixels.
        min_visibility: Configurable visibility threshold.

    Returns:
        True only when the point is in-frame and visible enough.
    """
    return is_inside_frame(x, y, frame_width, frame_height) and is_visible_enough(
        visibility,
        min_visibility,
    )


def describe_invalid_reason(
    keypoint: PixelKeypoint,
    min_visibility: float,
) -> str:
    """Explain why a landmark is hidden, without printing its coordinates.

    Args:
        keypoint: Validated pixel keypoint.
        min_visibility: Threshold used for the visibility check.

    Returns:
        A short reason such as ``out of frame`` or ``low visibility``.
    """
    reasons: list[str] = []
    if not keypoint.in_frame:
        reasons.append("out of frame")
    if not keypoint.visible:
        reasons.append(f"vis {keypoint.visibility:.2f}<{min_visibility:.2f}")
    if not reasons:
        return "invalid"
    return ", ".join(reasons)


def validate_pixel_keypoints(
    keypoints: list[PixelKeypoint],
    frame_width: int,
    frame_height: int,
    min_visibility: float,
) -> list[PixelKeypoint]:
    """Copy keypoints and set in-frame, visible, and valid flags.

    Raw x and y are preserved even when they are outside the image or below
    the visibility threshold. Nothing is clamped.

    Args:
        keypoints: Pixel keypoints from coordinate conversion.
        frame_width: Current frame width in pixels.
        frame_height: Current frame height in pixels.
        min_visibility: Minimum MediaPipe visibility that counts as valid.

    Returns:
        New PixelKeypoint objects with validation flags set.

    Raises:
        PoseDetectionError: If the frame size or threshold is invalid.
    """
    _require_frame_size(frame_width, frame_height)
    if not (0.0 <= min_visibility <= 1.0):
        raise PoseDetectionError(
            f"min_visibility must be in [0, 1], got {min_visibility}."
        )

    validated: list[PixelKeypoint] = []
    for keypoint in keypoints:
        in_frame = is_inside_frame(
            keypoint.x,
            keypoint.y,
            frame_width,
            frame_height,
        )
        visible = is_visible_enough(keypoint.visibility, min_visibility)
        validated.append(
            PixelKeypoint(
                name=keypoint.name,
                x=keypoint.x,
                y=keypoint.y,
                visibility=keypoint.visibility,
                in_frame=in_frame,
                visible=visible,
                is_valid=in_frame and visible,
            )
        )
    return validated


def valid_keypoints(keypoints: list[PixelKeypoint]) -> list[PixelKeypoint]:
    """Return only landmarks that passed in-frame and visibility checks.

    Args:
        keypoints: Validated pixel keypoints.

    Returns:
        Keypoints whose ``is_valid`` flag is True.
    """
    return [keypoint for keypoint in keypoints if keypoint.is_valid]


def count_valid(keypoints: list[PixelKeypoint]) -> int:
    """Count landmarks that are currently valid.

    Args:
        keypoints: Validated pixel keypoints.

    Returns:
        Number of keypoints with ``is_valid`` equal to True.
    """
    return sum(1 for keypoint in keypoints if keypoint.is_valid)


def _require_frame_size(frame_width: int, frame_height: int) -> None:
    """Reject non-positive frame dimensions.

    Args:
        frame_width: Current frame width in pixels.
        frame_height: Current frame height in pixels.

    Raises:
        PoseDetectionError: If width or height is not a positive pixel count.
    """
    if frame_width <= 0 or frame_height <= 0:
        raise PoseDetectionError(
            "Cannot validate pose coordinates because the frame size is "
            f"{frame_width}x{frame_height}."
        )
