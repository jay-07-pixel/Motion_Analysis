"""2D Motion Analysis using an Intel RealSense camera.

The live pipeline is:

    RealSense D455F -> RGB frame -> MediaPipe Pose -> raw pixels ->
    in-frame validation -> EMA smoothing -> display
"""

__version__ = "0.1.0"
