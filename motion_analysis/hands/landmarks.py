"""21 MediaPipe Hands landmarks, extraction, and pixel conversion.

MediaPipe Hands returns the same normalized image coordinates as Pose:
    pixel_x = normalized_x * frame_width
    pixel_y = normalized_y * frame_height

Frame width and height come from the current RGB image. Values are not
clamped; invalid landmarks are flagged later by the shared validator.
"""

from __future__ import annotations

from dataclasses import dataclass

from motion_analysis.hands.exceptions import HandsDetectionError
from motion_analysis.pose.landmarks import (
    NormalizedLandmark,
    PixelKeypoint,
    convert_to_pixel_coordinates,
)

# 21 landmarks per hand, in official MediaPipe Hands index order (0..20).
# Names match the model indices; INDEX/MIDDLE/RING drop the "_FINGER" infix
# used in some MediaPipe docs, but the index mapping is the same:
#   0 WRIST
#   1-4 THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP
#   5-8 INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP
#   9-12 MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP
#   13-16 RING_MCP, RING_PIP, RING_DIP, RING_TIP
#   17-20 PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP
HAND_LANDMARK_NAMES: tuple[str, ...] = (
    "WRIST",
    "THUMB_CMC",
    "THUMB_MCP",
    "THUMB_IP",
    "THUMB_TIP",
    "INDEX_MCP",
    "INDEX_PIP",
    "INDEX_DIP",
    "INDEX_TIP",
    "MIDDLE_MCP",
    "MIDDLE_PIP",
    "MIDDLE_DIP",
    "MIDDLE_TIP",
    "RING_MCP",
    "RING_PIP",
    "RING_DIP",
    "RING_TIP",
    "PINKY_MCP",
    "PINKY_PIP",
    "PINKY_DIP",
    "PINKY_TIP",
)

# Fingertips and wrist are listed in the overlay; all 21 are drawn on the image.
HAND_OVERLAY_NAMES: tuple[str, ...] = (
    "WRIST",
    "THUMB_TIP",
    "INDEX_TIP",
    "MIDDLE_TIP",
    "RING_TIP",
    "PINKY_TIP",
)

# Official MediaPipe Hands connections, stored as the named landmarks above.
HAND_CONNECTIONS: tuple[tuple[str, str], ...] = (
    ("WRIST", "THUMB_CMC"),
    ("THUMB_CMC", "INDEX_MCP"),
    ("INDEX_MCP", "MIDDLE_MCP"),
    ("MIDDLE_MCP", "RING_MCP"),
    ("RING_MCP", "PINKY_MCP"),
    ("WRIST", "PINKY_MCP"),
    ("THUMB_CMC", "THUMB_MCP"),
    ("THUMB_MCP", "THUMB_IP"),
    ("THUMB_IP", "THUMB_TIP"),
    ("INDEX_MCP", "INDEX_PIP"),
    ("INDEX_PIP", "INDEX_DIP"),
    ("INDEX_DIP", "INDEX_TIP"),
    ("MIDDLE_MCP", "MIDDLE_PIP"),
    ("MIDDLE_PIP", "MIDDLE_DIP"),
    ("MIDDLE_DIP", "MIDDLE_TIP"),
    ("RING_MCP", "RING_PIP"),
    ("RING_PIP", "RING_DIP"),
    ("RING_DIP", "RING_TIP"),
    ("PINKY_MCP", "PINKY_PIP"),
    ("PINKY_PIP", "PINKY_DIP"),
    ("PINKY_DIP", "PINKY_TIP"),
)


@dataclass(frozen=True)
class HandFrame:
    """One detected hand in pixel coordinates.

    Attributes:
        handedness: ``Left`` or ``Right`` from MediaPipe classification.
        score: Handedness confidence in [0, 1].
        frame_width: Width of the frame used for conversion, in pixels.
        frame_height: Height of the frame used for conversion, in pixels.
        normalized_landmarks: MediaPipe 0-1 coordinates for all 21 points.
        raw_keypoints: Unfiltered pixel keypoints. Drawn as red crosses.
        smoothed_keypoints: Filtered pixel keypoints. Drawn as cyan/green dots
            and used for motion measurements.
    """

    handedness: str
    score: float
    frame_width: int
    frame_height: int
    normalized_landmarks: list[NormalizedLandmark]
    raw_keypoints: list[PixelKeypoint]
    smoothed_keypoints: list[PixelKeypoint]


@dataclass(frozen=True)
class HandsFrame:
    """Hand detector output for a single RGB frame.

    Attributes:
        frame_width: Current frame width in pixels.
        frame_height: Current frame height in pixels.
        hands: Up to two hands (left and/or right).
    """

    frame_width: int
    frame_height: int
    hands: list[HandFrame]


def extract_hand_landmarks(hand_landmarks: object) -> list[NormalizedLandmark]:
    """Extract the 21 named landmarks from one MediaPipe hand.

    MediaPipe Hands returns landmarks in a fixed index order. This function
    assigns ``HAND_LANDMARK_NAMES[i]`` to landmark ``i`` so INDEX_TIP is
    always index 8, not a renamed nearby joint.

    The x and y values stay in MediaPipe's image-normalized space. They are
    not pixels and are not clamped.

    Args:
        hand_landmarks: Sequence of MediaPipe landmark objects, or an object
            with a ``landmark`` attribute.

    Returns:
        One NormalizedLandmark per hand point, in model index order.

    Raises:
        HandsDetectionError: If the result does not contain a landmark list.
    """
    raw_landmarks = getattr(hand_landmarks, "landmark", hand_landmarks)
    try:
        landmark_list = list(raw_landmarks)
    except TypeError as exc:
        raise HandsDetectionError(
            "MediaPipe Hands result has no landmark list to extract."
        ) from exc
    if not landmark_list:
        raise HandsDetectionError("MediaPipe Hands returned an empty landmark list.")

    extracted: list[NormalizedLandmark] = []
    for index, landmark in enumerate(landmark_list):
        name = (
            HAND_LANDMARK_NAMES[index]
            if index < len(HAND_LANDMARK_NAMES)
            else f"HAND_{index}"
        )
        visibility = getattr(landmark, "visibility", None)
        if visibility is None:
            visibility = getattr(landmark, "presence", None)
        # Hands often omit per-landmark visibility; a detected hand still
        # provides useful 2D finger points, so missing scores default to 1.
        if getattr(landmark, "x", None) is None or getattr(landmark, "y", None) is None:
            raise HandsDetectionError(
                f"MediaPipe Hands landmark {index} is missing x or y."
            )
        extracted.append(
            NormalizedLandmark(
                name=name,
                x=float(landmark.x),
                y=float(landmark.y),
                visibility=float(visibility) if visibility is not None else 1.0,
            )
        )
    return extracted


def hand_landmarks_to_pixels(
    landmarks: list[NormalizedLandmark],
    frame_width: int,
    frame_height: int,
) -> list[PixelKeypoint]:
    """Convert one hand's normalized landmarks to pixel keypoints.

    Uses the same conversion as body pose, with the size of the frame that
    will be displayed (no extra scale, crop, flip, or pad)::

        pixel_x = normalized_x * frame_width
        pixel_y = normalized_y * frame_height

    Args:
        landmarks: Normalized MediaPipe hand landmarks.
        frame_width: Current frame width in pixels from ``frame.shape[1]``.
        frame_height: Current frame height in pixels from ``frame.shape[0]``.

    Returns:
        PixelKeypoint list using the current frame size, not hard-coded
        resolution values. Out-of-frame estimates are kept, not clamped.
    """
    return convert_to_pixel_coordinates(landmarks, frame_width, frame_height)
