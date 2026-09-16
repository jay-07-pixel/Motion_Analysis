"""Load the official MediaPipe Hand Landmarker model."""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

from motion_analysis.hands.exceptions import HandsInitializationError

# Official Hand Landmarker bundle used for live RGB finger tracking.
HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
_MODEL_FILENAME = "hand_landmarker.task"


def hand_model_cache_path() -> Path:
    """Return the project-relative cache path for the Hand Landmarker model.

    Returns:
        Path under `motion_analysis/hands/_models/`.
    """
    return Path(__file__).resolve().parent / "_models" / _MODEL_FILENAME


def load_hand_landmarker_model() -> bytes:
    """Load Hand Landmarker model bytes, downloading them on first use.

    Returns:
        Raw `.task` model contents for `model_asset_buffer`.

    Raises:
        HandsInitializationError: If the model cannot be read or downloaded.
    """
    cache_path = hand_model_cache_path()
    if cache_path.is_file() and cache_path.stat().st_size > 0:
        return cache_path.read_bytes()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(HAND_LANDMARKER_MODEL_URL, timeout=60) as response:
            model_bytes = response.read()
    except urllib.error.URLError as exc:
        raise HandsInitializationError(
            "Could not download the MediaPipe Hand Landmarker model. "
            "Connect to the internet on the first run, then retry. "
            f"Original error: {exc}"
        ) from exc

    if not model_bytes:
        raise HandsInitializationError(
            "The MediaPipe Hand Landmarker download was empty."
        )

    cache_path.write_bytes(model_bytes)
    return model_bytes
