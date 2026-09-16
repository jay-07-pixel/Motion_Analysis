"""2D Motion Analysis using an Intel RealSense camera.

The live pipeline is:

    RealSense D455F or uploaded video -> RGB frame ->
    shared Pose + Hands pipeline -> 2D pixel keypoints ->
    position, displacement, and distance travelled
"""

__version__ = "0.1.0"
