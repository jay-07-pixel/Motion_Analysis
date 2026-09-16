# 2D Motion Analysis

Live RGB capture and 2D human pose detection for a motion analysis system built around an Intel RealSense D455f camera.

Current pipeline:

**RealSense D455F → RGB frame → MediaPipe Pose → raw pixel keypoints → validate in-frame → EMA smoothing → display smoothed skeleton**

The live overlay shows both **raw** and **smoothed** pixel coordinates so you can compare stability. Depth, 3D coordinates, velocity, acceleration, joint angles, trajectories, uploaded video, and a custom UI are not included yet.

## Project purpose

- Automatically detect a connected Intel RealSense camera
- Stream live RGB without hard-coded resolution or FPS
- Run MediaPipe Pose on each RGB frame
- Convert normalized MediaPipe landmarks into 2D pixel coordinates using the current frame size
- Validate those pixels against the live frame without clamping off-screen joints
- Smooth raw keypoints with a configurable EMA filter while keeping the raw values
- Draw the smoothed pose skeleton and a raw-vs-smoothed coordinate readout
- Shut down the camera, pose model, and OpenCV window cleanly

## Requirements

**Hardware**

- Intel RealSense D455f (other color-capable RealSense cameras also work)
- USB 3.0 port and USB 3 cable
- Display for the OpenCV preview window

**Software**

- Python 3.10 or newer (MediaPipe currently supports 3.10–3.12 best)
- Intel RealSense SDK 2.0 (Librealsense)
- Python packages listed in `requirements.txt`:
  - `pyrealsense2`
  - `opencv-python`
  - `numpy`
  - `mediapipe`

## Installation steps

1. Install [Python 3.10+](https://www.python.org/downloads/) and add it to your PATH.
2. Install the [Intel RealSense SDK 2.0](https://github.com/IntelRealSense/librealsense/releases) for your operating system. On Windows, use the official SDK installer so the camera drivers are present.
3. Connect the D455f to a USB 3 port and confirm Windows Device Manager (or the Intel RealSense Viewer) can see the camera.
4. Open a terminal in this project folder and create a virtual environment:

   ```bash
   python -m venv .venv
   ```

5. Activate the environment:

   - Windows (PowerShell): `.venv\Scripts\Activate.ps1`
   - Windows (Command Prompt): `.venv\Scripts\activate.bat`
   - macOS / Linux: `source .venv/bin/activate`

6. Install Python dependencies, including MediaPipe:

   ```bash
   pip install -r requirements.txt
   ```

The first time you run the program, it downloads Google's official Pose Landmarker model into `motion_analysis/pose/_models/` and reuses that cache afterward. No camera-specific or machine-specific file paths are hard-coded.

## How to run the program

From the project root, with the virtual environment activated:

```bash
python main.py
```

Or:

```bash
python -m motion_analysis
```

Press **Q** or **Esc** in the video window to stop capture and release the camera and pose detector. Closing the window also shuts everything down.

## How 2D keypoints are extracted

MediaPipe Pose does **not** return pixel coordinates. Each landmark comes back in **normalized image coordinates**:

- `x` is a fraction of the current image width, typically in `[0.0, 1.0]`
- `y` is a fraction of the current image height, typically in `[0.0, 1.0]`

Those values are stored as `NormalizedLandmark` objects and are only an intermediate MediaPipe format.

The motion-analysis system converts them with the **actual current frame size**:

```text
pixel_x = normalized_x * frame_width
pixel_y = normalized_y * frame_height
```

`frame_width` and `frame_height` are read from the live RGB frame (`frame.shape[1]` and `frame.shape[0]`). They are not hard-coded as 1280×720 or any other resolution.

The converted values are `PixelKeypoint` objects:

- `name`: landmark name, for example `LEFT_SHOULDER`
- `x`: pixel column in the current RGB frame
- `y`: pixel row in the current RGB frame
- `in_frame`: whether that pixel actually lies on the image

**Use smoothed `PixelKeypoint` values for display.** Raw values stay available on `PoseFrame.raw_keypoints`. Do not treat MediaPipe's normalized `x` and `y` as pixels.

## Why raw keypoints fluctuate

MediaPipe re-estimates every joint on every frame. Even if you stand still, the detector's confidence, lighting, compression, and small model updates move the landmark by a few pixels. That jitter is measurement noise, not real motion. Using those raw points directly would make later velocity or angle calculations look noisy.

## How coordinates are validated

MediaPipe's normalized `x` and `y` are **not guaranteed to stay in `[0, 1]`**. If a foot is cut off at the bottom of the camera, or the model infers a joint just beyond the image, `normalized_y` can be `1.05`. On a 720-pixel-tall frame that becomes:

```text
pixel_y = 1.05 * 720 = 756
```

which is greater than 720. That is not a conversion bug.

Visible pixels occupy `[0, width) x [0, height)`. A point with `y == 720` on a 720-tall image is already outside the last row (row index 719).

Validation therefore:

1. Converts with the **current** `frame.shape` width and height
2. Marks `in_frame = True` only when `0 <= x < width` and `0 <= y < height`
3. **Does not clamp** out-of-frame points onto the border, because clamping would invent a false on-screen joint
4. Skips out-of-frame landmarks when drawing the skeleton
5. By default, does not let out-of-frame raw samples update the smoother

The overlay labels these points `OUT y>=h` (or `x<0`, `x>=w`, `y<0`) next to the raw coordinate.

## What smoothing does

An exponential moving average (EMA) blends each new raw sample with the previous smoothed position:

```text
smoothed = alpha * raw + (1 - alpha) * previous_smoothed
```

- Smaller `alpha` → less jitter, more lag
- Larger `alpha` → follows motion faster, more residual noise
- Default `alpha = 0.35`

Change this by passing a `SmoothingConfig` into `PoseEstimator`. The values are not hard-coded inside the filter math.

Both series are kept:

- `PoseFrame.raw_keypoints` — unfiltered detector output
- `PoseFrame.smoothed_keypoints` — EMA output used for the cyan skeleton

If a raw joint is out of frame or low-visibility, the filter **holds** the last in-frame smoothed position instead of following the invalid estimate off-screen.

The live overlay prints both so you can compare stability:

```text
L.ANK  r=(610.2,742.8) OUT y>=h  s=(608.1,715.2) IN
```

`r=` is raw. `s=` is smoothed. Red dots on the video are raw in-frame joints; the cyan skeleton is smoothed.

The live overlay also still shows MediaPipe normalized values only as an explanation in the README; the on-screen panel now compares raw vs smoothed **pixels**.

## Expected output

The terminal prints the detected camera and confirms Pose is ready:

```text
Connected: Intel RealSense D455 (XXXXXXXX)
Resolution: 1280 x 720
Camera FPS: 30.0
MediaPipe Pose ready. Display uses smoothed pixel keypoints.
Smoothing: EMA alpha=0.35, hold_out_of_frame=True.
Press Q or Esc in the video window to quit.
```

An OpenCV window titled **2D Motion Analysis - Live RGB Pose** shows:

- The live RGB video
- Camera name, serial number, actual stream resolution, and FPS (top-left)
- The **smoothed** pose skeleton in cyan
- Raw in-frame joints as small red dots, so leftover jitter is visible
- A readout of useful body landmarks with **raw vs smoothed** pixel coordinates
- `OUT` labels when a raw estimate is outside the current frame
- `No person detected` when nobody is in view

Stand in front of the camera so the full body is visible. Shoulders, elbows, wrists, hips, knees, and ankles should track on the skeleton, and the overlay should list their pixel `(x, y)` values.

If no camera is connected, the program exits with a clear error instead of opening a blank window.

## Common camera errors and solutions

| Error | Likely cause | What to try |
| --- | --- | --- |
| `pyrealsense2 is not installed` | Python package or SDK missing | Run `pip install -r requirements.txt` and install the Intel RealSense SDK 2.0. |
| `No Intel RealSense camera with an RGB/color stream was detected` | Camera unplugged, USB 2 port, or driver issue | Use a USB 3 port and cable, reconnect the camera, and confirm it appears in Intel RealSense Viewer. |
| `Failed to initialize the RealSense RGB/color stream` | Another app already owns the camera | Close Intel RealSense Viewer, browser camera tabs, and other capture apps, then rerun. |
| `Timed out waiting for a RealSense color frame` | Cable, USB bandwidth, or disconnect during capture | Switch to a USB 3 port, avoid hubs if possible, and reconnect the camera. |
| `device disappeared before the stream could start` | Unstable connection during start-up | Replug the camera, wait a few seconds, then run again. |
| OpenCV window never appears | Headless session or missing GUI libraries | Run on a desktop session with a display. On Linux, install OpenCV GUI dependencies for your distro. |
| Colors look wrong or the image is tinted | Unusual stream format | Reconnect the camera and restart. The app converts RGB/YUYV to BGR when the SDK reports those formats. |
| Low FPS overlay compared with Camera FPS | CPU load, pose model, or USB 2 fallback | Close extra programs and confirm the USB connection is 3.0. Display FPS is measured independently of the camera profile. |

## Common pose errors and solutions

| Error | Likely cause | What to try |
| --- | --- | --- |
| `MediaPipe is not installed` | Pose dependency missing | Run `pip install -r requirements.txt` inside the virtual environment. |
| `Could not download the MediaPipe Pose Landmarker model` | First run has no internet | Connect to the internet once so the official `.task` model can be cached, then rerun. |
| Overlay shows `No person detected` | Person too close, too far, or poorly lit | Step back so more of the body is visible and add more light. |
| Overlay shows `OUT y>=h` or similar | Joint is off-screen or inferred beyond the image | Step back so the full body fits. The estimate is kept but not clamped or drawn as an on-screen joint. |
| Skeleton jitter or missing limbs | Occlusion, low visibility, or raw (unsmoothed) noise | Face the camera and keep joints in view. Lower `SmoothingConfig.alpha` for more filtering. |
| Coordinates look wrong for the image size | Using normalized values as pixels | Use the `px=` values. `norm=` is MediaPipe's 0–1 space and is not a pixel location. |

After any camera or pose initialization error the program stops the RealSense pipeline, closes MediaPipe Pose, and closes OpenCV windows so the device is not left locked.
