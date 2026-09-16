"""Selectable body and hand landmarks for 2D motion measurements.

Landmark names come from the MediaPipe Pose and Hands catalogs already used
by detection. Pixel positions are never hard-coded; only names and groups
are listed so the user can choose what to analyse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from motion_analysis.hands.landmarks import HAND_LANDMARK_NAMES, HandsFrame
from motion_analysis.pose.landmarks import POSE_LANDMARK_NAMES, PixelKeypoint, PoseFrame

BODY_GROUP = "Body"
LEFT_HAND_GROUP = "Left Hand"
RIGHT_HAND_GROUP = "Right Hand"

BODY_PREFIX = "BODY"
LEFT_HAND_PREFIX = "LEFT_HAND"
RIGHT_HAND_PREFIX = "RIGHT_HAND"

# Default analysis target: a named body joint, not a hard-coded pixel.
DEFAULT_LANDMARK_KEY = f"{BODY_PREFIX}:LEFT_WRIST"


@dataclass(frozen=True)
class LandmarkTarget:
    """One user-selectable landmark.

    Attributes:
        key: Stable id such as ``BODY:LEFT_WRIST`` or ``LEFT_HAND:INDEX_TIP``.
        group: Display group (Body, Left Hand, Right Hand).
        name: MediaPipe landmark name within that group.
    """

    key: str
    group: str
    name: str

    @property
    def label(self) -> str:
        """Human-readable label for menus and overlays."""
        return f"{self.group} · {self.name}"


@dataclass(frozen=True)
class KeypointObservation:
    """Raw and smoothed pixels for one landmark on one frame.

    Attributes:
        smoothed: Filtered pixel keypoint used for motion math.
        raw: Unfiltered detector pixels, kept for comparison.
    """

    smoothed: PixelKeypoint
    raw: PixelKeypoint


def selectable_landmarks() -> tuple[LandmarkTarget, ...]:
    """Build the full list of body and hand landmarks the user may analyse.

    Returns:
        Pose landmarks first, then left-hand, then right-hand points.
    """
    body = tuple(
        LandmarkTarget(key=f"{BODY_PREFIX}:{name}", group=BODY_GROUP, name=name)
        for name in POSE_LANDMARK_NAMES
    )
    left = tuple(
        LandmarkTarget(
            key=f"{LEFT_HAND_PREFIX}:{name}",
            group=LEFT_HAND_GROUP,
            name=name,
        )
        for name in HAND_LANDMARK_NAMES
    )
    right = tuple(
        LandmarkTarget(
            key=f"{RIGHT_HAND_PREFIX}:{name}",
            group=RIGHT_HAND_GROUP,
            name=name,
        )
        for name in HAND_LANDMARK_NAMES
    )
    return body + left + right


def default_landmark() -> LandmarkTarget:
    """Return the default analysis landmark.

    Returns:
        ``BODY:LEFT_WRIST`` when present, otherwise the first catalog entry.
    """
    catalog = selectable_landmarks()
    for target in catalog:
        if target.key == DEFAULT_LANDMARK_KEY:
            return target
    return catalog[0]


def landmark_by_key(key: str) -> LandmarkTarget:
    """Look up a catalog entry by key.

    Args:
        key: Id such as ``BODY:NOSE``.

    Returns:
        Matching LandmarkTarget.

    Raises:
        KeyError: If the key is not in the catalog.
    """
    for target in selectable_landmarks():
        if target.key == key:
            return target
    raise KeyError(f"Unknown landmark key: {key}")


def cycle_landmark(current: LandmarkTarget, step: int) -> LandmarkTarget:
    """Move to the next or previous selectable landmark.

    Args:
        current: Landmark currently being analysed.
        step: +1 for next, -1 for previous. Other integers wrap the catalog.

    Returns:
        The landmark at the wrapped index.
    """
    catalog = selectable_landmarks()
    keys = [target.key for target in catalog]
    try:
        index = keys.index(current.key)
    except ValueError:
        index = 0
    return catalog[(index + step) % len(catalog)]


def collect_observations(
    pose_frame: Optional[PoseFrame],
    hands_frame: Optional[HandsFrame],
) -> dict[str, KeypointObservation]:
    """Map catalog keys to this frame's raw and smoothed keypoints.

    Missing pose or missing hands simply omit those keys. The motion tracker
    treats omitted keys as invalid for this frame.

    Args:
        pose_frame: Body result, or None if nobody was detected.
        hands_frame: Hands result, or None if detection failed.

    Returns:
        Observations keyed by LandmarkTarget.key.
    """
    observations: dict[str, KeypointObservation] = {}
    if pose_frame is not None:
        raw_by_name = {point.name: point for point in pose_frame.raw_keypoints}
        for smoothed in pose_frame.smoothed_keypoints:
            raw = raw_by_name.get(smoothed.name, smoothed)
            observations[f"{BODY_PREFIX}:{smoothed.name}"] = KeypointObservation(
                smoothed=smoothed,
                raw=raw,
            )
    if hands_frame is not None:
        for hand in hands_frame.hands:
            prefix = _hand_prefix(hand.handedness)
            if prefix is None:
                continue
            raw_by_name = {point.name: point for point in hand.raw_keypoints}
            for smoothed in hand.smoothed_keypoints:
                raw = raw_by_name.get(smoothed.name, smoothed)
                observations[f"{prefix}:{smoothed.name}"] = KeypointObservation(
                    smoothed=smoothed,
                    raw=raw,
                )
    return observations


def _hand_prefix(handedness: str) -> Optional[str]:
    """Map MediaPipe handedness text to a catalog prefix.

    Args:
        handedness: Classification label such as ``Left`` or ``Right``.

    Returns:
        Catalog prefix, or None if the label is not left/right.
    """
    label = handedness.strip().lower()
    if label == "left":
        return LEFT_HAND_PREFIX
    if label == "right":
        return RIGHT_HAND_PREFIX
    return None
