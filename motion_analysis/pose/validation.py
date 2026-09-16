"""Validate 2D pixel keypoints against the live frame size.

MediaPipe Pose does not guarantee normalized x/y stay in [0, 1]. When a joint
is off-screen or inferred beyond the image, conversion produces pixel values
outside the visible frame, for example y >= 720 on a 720-pixel-tall image.

This module flags those landmarks. It does not clamp them onto the image
border, which would invent a false on-screen position.
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


def describe_frame_position(
    x: float,
    y: float,
    frame_width: int,
    frame_height: int,
) -> str:
    """Describe why a coordinate is inside or outside the frame.

    Args:
        x: Horizontal pixel coordinate.
        y: Vertical pixel coordinate.
        frame_width: Current frame width in pixels.
        frame_height: Current frame height in pixels.

    Returns:
        A short status string such as ``IN``, ``OUT x<0``, or ``OUT y>=h``.
    """
    if not math.isfinite(x) or not math.isfinite(y):
        return "OUT non-finite"
    reasons: list[str] = []
    if x < 0.0:
        reasons.append("x<0")
    elif x >= frame_width:
        reasons.append("x>=w")
    if y < 0.0:
        reasons.append("y<0")
    elif y >= frame_height:
        reasons.append("y>=h")
    if not reasons:
        return "IN"
    return "OUT " + ",".join(reasons)


def validate_pixel_keypoints(
    keypoints: list[PixelKeypoint],
    frame_width: int,
    frame_height: int,
) -> list[PixelKeypoint]:
    """Copy keypoints and set ``in_frame`` from the current frame size.

    Raw x and y are preserved even when they are outside the image.

    Args:
        keypoints: Pixel keypoints from coordinate conversion.
        frame_width: Current frame width in pixels.
        frame_height: Current frame height in pixels.

    Returns:
        New PixelKeypoint objects with validated ``in_frame`` flags.

    Raises:
        PoseDetectionError: If the frame size is invalid.
    """
    _require_frame_size(frame_width, frame_height)
    validated: list[PixelKeypoint] = []
    for keypoint in keypoints:
        validated.append(
            PixelKeypoint(
                name=keypoint.name,
                x=keypoint.x,
                y=keypoint.y,
                visibility=keypoint.visibility,
                in_frame=is_inside_frame(
                    keypoint.x,
                    keypoint.y,
                    frame_width,
                    frame_height,
                ),
            )
        )
    return validated


def count_out_of_frame(keypoints: list[PixelKeypoint]) -> int:
    """Count landmarks whose pixel coordinates are outside the image.

    Args:
        keypoints: Validated pixel keypoints.

    Returns:
        Number of keypoints with ``in_frame`` equal to False.
    """
    return sum(1 for keypoint in keypoints if not keypoint.in_frame)


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
