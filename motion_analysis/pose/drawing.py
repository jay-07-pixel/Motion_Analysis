"""Drawing helpers for the live 2D pose overlay."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from motion_analysis.pose.landmarks import (
    TEST_LANDMARK_NAMES,
    PixelKeypoint,
    PoseFrame,
    pixel_to_draw_xy,
)
from motion_analysis.pose.smoothing import SmoothingConfig
from motion_analysis.pose.validation import count_valid, describe_invalid_reason

# Official MediaPipe Pose skeleton edges, stored as landmark names.
POSE_SKELETON_CONNECTIONS: tuple[tuple[str, str], ...] = (
    ("NOSE", "LEFT_EYE_INNER"),
    ("LEFT_EYE_INNER", "LEFT_EYE"),
    ("LEFT_EYE", "LEFT_EYE_OUTER"),
    ("LEFT_EYE_OUTER", "LEFT_EAR"),
    ("NOSE", "RIGHT_EYE_INNER"),
    ("RIGHT_EYE_INNER", "RIGHT_EYE"),
    ("RIGHT_EYE", "RIGHT_EYE_OUTER"),
    ("RIGHT_EYE_OUTER", "RIGHT_EAR"),
    ("MOUTH_LEFT", "MOUTH_RIGHT"),
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_SHOULDER", "LEFT_ELBOW"),
    ("LEFT_ELBOW", "LEFT_WRIST"),
    ("LEFT_WRIST", "LEFT_PINKY"),
    ("LEFT_WRIST", "LEFT_INDEX"),
    ("LEFT_WRIST", "LEFT_THUMB"),
    ("LEFT_PINKY", "LEFT_INDEX"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
    ("RIGHT_ELBOW", "RIGHT_WRIST"),
    ("RIGHT_WRIST", "RIGHT_PINKY"),
    ("RIGHT_WRIST", "RIGHT_INDEX"),
    ("RIGHT_WRIST", "RIGHT_THUMB"),
    ("RIGHT_PINKY", "RIGHT_INDEX"),
    ("LEFT_SHOULDER", "LEFT_HIP"),
    ("RIGHT_SHOULDER", "RIGHT_HIP"),
    ("LEFT_HIP", "RIGHT_HIP"),
    ("LEFT_HIP", "LEFT_KNEE"),
    ("LEFT_KNEE", "LEFT_ANKLE"),
    ("LEFT_ANKLE", "LEFT_HEEL"),
    ("LEFT_ANKLE", "LEFT_FOOT_INDEX"),
    ("LEFT_HEEL", "LEFT_FOOT_INDEX"),
    ("RIGHT_HIP", "RIGHT_KNEE"),
    ("RIGHT_KNEE", "RIGHT_ANKLE"),
    ("RIGHT_ANKLE", "RIGHT_HEEL"),
    ("RIGHT_ANKLE", "RIGHT_FOOT_INDEX"),
    ("RIGHT_HEEL", "RIGHT_FOOT_INDEX"),
)

_SMOOTHED_SKELETON_COLOR = (0, 255, 255)
_SMOOTHED_JOINT_COLOR = (0, 215, 255)
_LABEL_COLOR = (255, 255, 255)


def draw_pose_skeleton(
    frame: np.ndarray,
    keypoints: list[PixelKeypoint],
) -> np.ndarray:
    """Draw the pose skeleton from currently valid pixel keypoints.

    A bone is drawn only when both endpoints are valid. Invalid joints are
    skipped instead of being clamped onto the image edge.

    Args:
        frame: BGR image to draw on. Modified in place.
        keypoints: 2D landmarks already converted to pixel coordinates.

    Returns:
        The same frame, with joints and bones drawn.
    """
    points = {keypoint.name: keypoint for keypoint in keypoints if keypoint.is_valid}
    for start_name, end_name in POSE_SKELETON_CONNECTIONS:
        start = points.get(start_name)
        end = points.get(end_name)
        if start is None or end is None:
            continue
        cv2.line(
            frame,
            _as_drawing_point(start),
            _as_drawing_point(end),
            _SMOOTHED_SKELETON_COLOR,
            2,
            cv2.LINE_AA,
        )
    for keypoint in points.values():
        cv2.circle(
            frame,
            _as_drawing_point(keypoint),
            5,
            _SMOOTHED_JOINT_COLOR,
            -1,
            cv2.LINE_AA,
        )
    return frame


def draw_landmark_coordinates(
    frame: np.ndarray,
    pose_frame: PoseFrame,
    landmark_names: tuple[str, ...] = TEST_LANDMARK_NAMES,
    smoothing: Optional[SmoothingConfig] = None,
) -> np.ndarray:
    """Overlay valid smoothed pixel coordinates for live testing.

    Invalid landmarks are listed by reason only. Their out-of-frame or
    low-confidence coordinates are not shown.

    Args:
        frame: BGR image to draw on. Modified in place.
        pose_frame: Detection result with raw and smoothed keypoints.
        landmark_names: Subset of body landmarks to print on screen.
        smoothing: Filter settings shown in the overlay header.

    Returns:
        The same frame, with a coordinate readout panel.
    """
    raw_by_name = {keypoint.name: keypoint for keypoint in pose_frame.raw_keypoints}
    smooth_by_name = {
        keypoint.name: keypoint for keypoint in pose_frame.smoothed_keypoints
    }
    valid_count = count_valid(pose_frame.smoothed_keypoints)
    total = len(pose_frame.smoothed_keypoints)
    min_visibility = smoothing.min_visibility if smoothing is not None else 0.5
    alpha = smoothing.alpha if smoothing is not None else float("nan")

    lines = [
        "Valid smoothed 2D pixels only",
        f"Frame {pose_frame.frame_width}x{pose_frame.frame_height}  "
        f"vis>={min_visibility:.2f}  EMA a={alpha:.2f}",
        f"Valid joints: {valid_count}/{total}",
    ]
    for name in landmark_names:
        smooth = smooth_by_name.get(name)
        if smooth is None:
            lines.append(f"{_short_name(name)}: not detected")
            continue
        if not smooth.is_valid:
            raw = raw_by_name.get(name, smooth)
            reason = describe_invalid_reason(raw, min_visibility)
            lines.append(f"{_short_name(name)}: hidden ({reason})")
            continue
        lines.append(
            f"{_short_name(name)}  "
            f"({smooth.x:.1f}, {smooth.y:.1f})  vis={smooth.visibility:.2f}"
        )

    _draw_text_panel(frame, lines, top_right=True)
    return frame


def draw_pose_on_frame(
    frame: np.ndarray,
    pose_frame: Optional[PoseFrame],
    error_message: Optional[str] = None,
    smoothing: Optional[SmoothingConfig] = None,
) -> np.ndarray:
    """Copy a camera frame and add pose drawing for display.

    Args:
        frame: Live BGR frame from the RealSense RGB stream.
        pose_frame: Pixel-space pose result, or None if no person was found.
        error_message: Optional per-frame pose error to show instead of
            landmark coordinates.
        smoothing: Filter settings included in the overlay.

    Returns:
        Annotated copy of the input frame. The original camera frame is not
        modified.
    """
    canvas = frame.copy()
    if error_message:
        _draw_text_panel(canvas, ["Pose error", error_message], top_right=True)
        return canvas
    if pose_frame is None:
        _draw_text_panel(canvas, ["No person detected"], top_right=True)
        return canvas

    draw_pose_skeleton(canvas, pose_frame.smoothed_keypoints)
    draw_landmark_coordinates(canvas, pose_frame, smoothing=smoothing)
    return canvas


def _short_name(name: str) -> str:
    """Abbreviate a landmark name for the overlay.

    Args:
        name: Full MediaPipe landmark name.

    Returns:
        A shorter label that still identifies left/right and the joint.
    """
    return (
        name.replace("LEFT_", "L.")
        .replace("RIGHT_", "R.")
        .replace("SHOULDER", "SH")
        .replace("ELBOW", "ELB")
        .replace("WRIST", "WR")
        .replace("ANKLE", "ANK")
        .replace("KNEE", "KN")
    )


def _as_drawing_point(keypoint: PixelKeypoint) -> tuple[int, int]:
    """Integer image pixel for this keypoint, matching printed ``drawn`` coords.

    Args:
        keypoint: Landmark already expressed in pixels.

    Returns:
        Integer (x, y) suitable for cv2.line and cv2.circle.
    """
    return pixel_to_draw_xy(keypoint.x, keypoint.y)


def _draw_text_panel(
    frame: np.ndarray,
    lines: list[str],
    top_right: bool = False,
) -> None:
    """Draw a semi-transparent text panel onto a frame.

    Args:
        frame: BGR image modified in place.
        lines: Text rows to display.
        top_right: If True, place the panel on the top-right; otherwise
            top-left.
    """
    padding = 8
    line_height = 17
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.40
    thickness = 1
    text_sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    box_width = max(size[0] for size in text_sizes) + padding * 2
    box_height = line_height * len(lines) + padding * 2
    x0 = frame.shape[1] - box_width - 8 if top_right else 8
    y0 = 8
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x0, y0),
        (x0 + box_width, y0 + box_height),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    y = y0 + padding + 12
    for line in lines:
        cv2.putText(
            frame,
            line,
            (x0 + padding, y),
            font,
            scale,
            _LABEL_COLOR,
            thickness,
            cv2.LINE_AA,
        )
        y += line_height
