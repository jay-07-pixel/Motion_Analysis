"""Load the official MediaPipe Pose Landmarker model.

The Tasks API needs the `.task` model in memory. This module downloads it
from Google's public model catalog on first use and caches it next to this
package. No machine-specific paths are hard-coded.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

from motion_analysis.pose.exceptions import PoseInitializationError

# Official lite Pose Landmarker bundle used for live RGB inference.
POSE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
_MODEL_FILENAME = "pose_landmarker_lite.task"


def pose_model_cache_path() -> Path:
    """Return the project-relative cache path for the Pose Landmarker model.

    Returns:
        Path under `motion_analysis/pose/_models/`. This stays inside the
        project rather than using a user-specific absolute path.
    """
    return Path(__file__).resolve().parent / "_models" / _MODEL_FILENAME


def load_pose_landmarker_model() -> bytes:
    """Load Pose Landmarker model bytes, downloading them on first use.

    Returns:
        Raw `.task` model contents for `model_asset_buffer`.

    Raises:
        PoseInitializationError: If the model cannot be read or downloaded.
    """
    cache_path = pose_model_cache_path()
    if cache_path.is_file() and cache_path.stat().st_size > 0:
        return cache_path.read_bytes()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(POSE_LANDMARKER_MODEL_URL, timeout=60) as response:
            model_bytes = response.read()
    except urllib.error.URLError as exc:
        raise PoseInitializationError(
            "Could not download the MediaPipe Pose Landmarker model. "
            "Connect to the internet on the first run, then retry. "
            f"Original error: {exc}"
        ) from exc

    if not model_bytes:
        raise PoseInitializationError(
            "The MediaPipe Pose Landmarker download was empty."
        )

    cache_path.write_bytes(model_bytes)
    return model_bytes
