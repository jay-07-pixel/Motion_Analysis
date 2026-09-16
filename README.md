# 2D Motion Analysis

Live RGB capture or uploaded-video analysis with 2D body pose and 21-point hand landmarks, built around an Intel RealSense D455f camera.

Current pipeline:

```text
Live D455F ─┐
            ├──→ Common Frame Processing → 2D Keypoints (x, y)
Video File ─┘                │
                             └──→ Position, Displacement, Distance travelled
```

The shared path after a frame is available is always:

**RGB → Pose + Hands → validation → smoothing → 2D pixel keypoints → 2D motion measurements**

Live camera and uploaded video use that path identically. Depth, 3D, velocity, acceleration, joint angles, and trajectories are not included yet.

## Project purpose

- Choose **Live Camera** (RealSense D455F) or **Upload Video**
- Stream live RGB or read an uploaded file without hard-coded paths, resolution, or FPS
- Run the same MediaPipe Pose and Hands pipeline on every RGB frame
- Convert body and hand landmarks into 2D pixel coordinates using the current frame size
- Treat a landmark as valid only when it is inside the live frame and MediaPipe visibility passes a threshold
- Smooth valid keypoints with a configurable EMA filter while keeping raw values internally
- Draw the body skeleton plus all visible finger landmarks and connections
- Let the user pick a body or hand landmark and measure its 2D motion
- Shut down the camera, pose model, and OpenCV window cleanly

## Requirements

**Hardware**

- Intel RealSense D455f (other color-capable RealSense cameras also work)
- USB 3.0 port and USB 3 cable
- Display for the OpenCV preview window

**Software**

- Python 3.10 or newer (MediaPipe currently supports 3.10–3.12 best)
- Intel RealSense SDK 2.0 (Librealsense) for live camera mode
- Tkinter (included with most Python installers; used for the start menu and file picker)
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

The first time you run the program, it downloads Google's official Pose Landmarker and Hand Landmarker models into `motion_analysis/pose/_models/` and `motion_analysis/hands/_models/`, then reuses those caches. No camera-specific or machine-specific file paths are hard-coded.

## How to run the program

From the project root, with the virtual environment activated:

```bash
python main.py
```

Or:

```bash
python -m motion_analysis
```

A start screen offers two buttons:

- **Live Camera** — detect the connected RealSense D455F and process its RGB stream
- **Upload Video** — open a file picker and process the selected video

After you choose an input, a landmark list appears. Pick the body or hand joint to analyse (default `LEFT_WRIST`). You can change it later with **[** and **]** in the video window.

No video path is hard-coded. Closing the start screen exits the program. After a session ends, the start screen returns so you can switch inputs.

### Live Camera

- Requires a connected D455F on USB 3
- Press **Q** or **Esc** to stop and return to the start screen

### Upload Video

1. Click **Upload Video** and choose a file (MP4, AVI, MOV, MKV, and similar)
2. The app reads that file's actual width, height, FPS, frame count, and duration
3. Controls:
   - **Space** — pause / play
   - **R** — restart from the first frame
   - **Q** or **Esc** — exit to the start screen

The overlay shows the current frame number, time, source FPS, and play/pause state. At the end of the file, playback pauses on the last frame; press **R** to restart. Restart also clears smoothing history and motion totals.

Both modes show the selected landmark, its current smoothed position, displacement from the first valid sample, and distance travelled.

## How 2D keypoints are extracted

### MediaPipe normalized coordinates

MediaPipe Pose and MediaPipe Hands do **not** return pixel coordinates. Each landmark comes back in **normalized image coordinates**:

- `x` is a fraction of the current image width
- `y` is a fraction of the current image height
- those values are often in `[0.0, 1.0]`, but they can go outside that range

Normalized `x` and `y` are stored as `NormalizedLandmark` objects. They are an intermediate MediaPipe format, not the coordinates used by this motion-analysis system.

### Conversion to pixel coordinates

The system converts with the **actual current frame size**:

```text
pixel_x = normalized_x * frame_width
pixel_y = normalized_y * frame_height
```

`frame_width` and `frame_height` come from the live RGB frame (`frame.shape[1]` and `frame.shape[0]`). They are never hard-coded as 1280×720 or any other resolution. Live camera and uploaded video both use this conversion on whatever size the current frame actually has.

The converted values are `PixelKeypoint` objects with:

- `name`: landmark name, for example `LEFT_SHOULDER`
- `x`, `y`: pixel coordinates
- `visibility`: MediaPipe visibility/confidence
- `is_valid`: whether the landmark may be drawn and used for smoothing

Raw converted values are kept on `PoseFrame.raw_keypoints`. Smoothed values are kept on `PoseFrame.smoothed_keypoints`.

### Landmark visibility/confidence

MediaPipe also reports a **visibility** score in `[0, 1]`. This is the model's confidence that the joint is present and unoccluded.

A landmark is **valid** only when both checks pass:

1. Pixel coordinates are inside the current frame: `0 <= x < width` and `0 <= y < height`
2. `visibility >= min_visibility` (default `0.50`, set on `SmoothingConfig`)

Low-visibility joints are treated as invalid even if their predicted `(x, y)` sits inside the image. That stops occluded or guessed limbs from being drawn.

### Why out-of-frame landmarks are rejected

If a foot is cut off at the bottom of the camera, or the model infers a joint just beyond the image, `normalized_y` can be `1.05`. On a 720-pixel-tall frame:

```text
pixel_y = 1.05 * 720 = 756
```

which is outside the image. Visible pixels occupy `[0, width) x [0, height)`, so `y == 720` is already past the last row.

Those estimates are **not clamped** onto the border. Clamping would invent a false on-screen joint and make the skeleton stick to the edge. Instead the landmark is marked invalid, omitted from the drawing, and omitted from the on-screen coordinate list.

### Why smoothing is used

MediaPipe re-estimates every joint on every frame. Even if you stand still, detector noise moves the landmark by a few pixels. That jitter is not real motion.

An exponential moving average (EMA) blends each **valid** raw sample with the previous smoothed position:

```text
smoothed = alpha * raw + (1 - alpha) * previous_smoothed
```

- Smaller `alpha` → less jitter, more lag
- Larger `alpha` → follows motion faster, more residual noise
- Default `alpha = 0.35`, set on `SmoothingConfig`

Smoothing runs **only on valid observations**. If a landmark is briefly out of frame or low-visibility, the last valid smoothed position is held internally and **not drawn**, so the skeleton does not jump or fake a movement. When the joint becomes valid again, filtering resumes from that held state.

The overlay prints only valid smoothed pixels, for example:

```text
L.SH  (502.3, 210.1)  vis=0.97
L.ANK: hidden (out of frame)
```

## Hand and finger landmarks (21 per hand)

MediaPipe Hands runs on the same live RGB frame as body pose. It can detect **both left and right hands**. Each detected hand has **21 landmarks**:

| Name | Meaning |
| --- | --- |
| `WRIST` | Wrist |
| `THUMB_CMC`, `THUMB_MCP`, `THUMB_IP`, `THUMB_TIP` | Thumb base to tip |
| `INDEX_MCP`, `INDEX_PIP`, `INDEX_DIP`, `INDEX_TIP` | Index finger |
| `MIDDLE_MCP`, `MIDDLE_PIP`, `MIDDLE_DIP`, `MIDDLE_TIP` | Middle finger |
| `RING_MCP`, `RING_PIP`, `RING_DIP`, `RING_TIP` | Ring finger |
| `PINKY_MCP`, `PINKY_PIP`, `PINKY_DIP`, `PINKY_TIP` | Little finger |

Those names are stored on each `PixelKeypoint` and drawn next to the joint on the video.

### 2D pixel-coordinate extraction for hands

Hand landmarks use the same conversion as the body:

```text
pixel_x = normalized_x * frame_width
pixel_y = normalized_y * frame_height
```

`frame_width` and `frame_height` come from the current RGB frame. They are not hard-coded.

Each hand keeps:

- `raw_keypoints` — unfiltered detector pixels
- `smoothed_keypoints` — EMA output used for drawing

A finger landmark is drawn only when it is inside the live frame and passes the visibility/confidence threshold. Out-of-frame points are **not clamped**. If a fingertip is briefly lost, smoothing holds the last valid position internally and does not draw a fake jump.

The live overlay lists wrist and fingertip pixels for each hand, for example:

```text
Left  21/21  score=0.97
  INDEX_TIP  (812.4, 410.2)  vis=1.00
Right  18/21  score=0.91
  PINKY_TIP: hidden (out of frame)
```

## 2D motion measurements

After smoothing, every valid tracked keypoint is measured in **pixel coordinates**. The overlay shows the landmark you selected; internally the same formulas run for all body and hand joints.

Calculations use **smoothed** `(x, y)` only. Raw detector pixels stay available internally for comparison. They are not used for displacement or distance travelled.

### Formulas

**Position** is the current smoothed location:

```text
P = (x, y)
```

**Displacement** is the straight-line change from the first valid sample of that landmark, not the path length:

```text
ΔP = P_current - P_initial
ΔP = (x_current - x_initial, y_current - y_initial)
```

**2D displacement magnitude**:

```text
|ΔP| = sqrt((x_current - x_initial)^2 + (y_current - y_initial)^2)
```

**Distance travelled** is the sum of movement between consecutive valid frames:

```text
d_i = sqrt((x_i - x_{i-1})^2 + (y_i - y_{i-1})^2)
distance travelled = d_1 + d_2 + ... + d_n
```

`P_initial` is the first valid smoothed position after the session starts (or after a video restart). Distance travelled starts at `0` on that sample.

### Invalid and missing landmarks

A landmark must be valid on the current frame before movement is calculated: in-frame and above the visibility threshold, using the same rules as skeleton drawing.

- If the joint is missing, out of frame, or low-confidence, **position and displacement are not calculated** for that frame.
- Distance travelled is **not increased**. A gap does not add a teleport from the last seen location.
- When the joint becomes valid again, tracking resumes from the new position. Held path length is unchanged across the gap.

### Timestamps and FPS

Frame time comes from the active source:

- Live D455F: RealSense color-frame timestamp when the SDK provides one
- Uploaded video: file playback position, or `frame_index / source_FPS` when the header reports FPS
- If the source has no clock, a monotonic timer is used instead of assuming 30 FPS

Source FPS is read from the live stream profile or the video file header. It is never hard-coded. The motion panel prints that FPS next to the frame timestamp.

### Selecting a landmark

1. After **Live Camera** or **Upload Video**, choose a joint from the list (Body, Left Hand, Right Hand).
2. During a session, press **[** or **,** for the previous landmark and **]** or **.** for the next.
3. The magenta ring marks the joint being measured. Switching the overlay target does not reset other joints' totals. **R** on an uploaded video restarts the file and clears all motion totals.

The bottom-left overlay shows:

```text
Landmark: Body · LEFT_WRIST
t = 3.210s   source FPS = 30.00
Position P = (502.3, 210.1) px
Displacement ΔP = (-12.4, 8.1) px
|ΔP| = 14.8 px
Distance travelled = 203.5 px
```

Velocity, acceleration, angles, and 3D are not computed in this step.



## Expected output

The terminal prints the detected camera and confirms Pose is ready:

```text
Select input
[Live Camera]  [Upload Video]
```

For live camera:

```text
Source: Intel RealSense D455 (XXXXXXXX)
Resolution: 1280 x 720
Source FPS: 30.0
MediaPipe Pose and Hands ready. Both inputs share one 2D pipeline.
Press Q or Esc in the video window to quit.
```

For an uploaded video, resolution and FPS come from the file header (not 1280×720). The window also shows frame number, time, and **Space / R / Q** controls.

Both modes draw the same overlays: body skeleton, 21-point hands, valid 2D pixel coordinates, and the selected landmark's position, displacement, and distance travelled.

Stand in front of the camera so the full body is visible, and hold your hands in view for finger tracking. For uploaded video, keep the person and hands clearly in shot.

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
| `Could not download the MediaPipe Hand Landmarker model` | First run has no internet | Same as above; the hands model is cached under `motion_analysis/hands/_models/`. |
| Could not open the selected video file | Unsupported codec or corrupt file | Try MP4 (H.264) or another OpenCV-readable format. |
| Overlay shows `Source FPS: unknown` | The file header did not report FPS | Playback still runs; frame numbers still update. |
| Overlay shows `No person detected` | Person too close, too far, or poorly lit | Step back so more of the body is visible and add more light. |
| Overlay shows `hidden (out of frame)` | Joint is off-screen or inferred beyond the image | Step back so the full body fits. The estimate is kept internally but not drawn or clamped. |
| Overlay shows `hidden (vis …)` | MediaPipe visibility is below the threshold | Improve lighting, face the camera, and keep the joint unoccluded. Lower `SmoothingConfig.min_visibility` only if needed. |
| Skeleton jitter or missing limbs | Occlusion, low visibility, or residual detector noise | Keep joints in view. Lower `SmoothingConfig.alpha` for more filtering. |
| Coordinates look wrong for the image size | Using MediaPipe normalized values as pixels | Use the on-screen valid pixel `(x, y)` values. Normalized MediaPipe coordinates are not pixels. |
| Overlay shows `Position: not calculated` | Selected joint is missing, out of frame, or low-visibility | Bring that landmark into view. Distance travelled is held, not increased. |
| Displacement stays near 0 while distance grows | The joint returned close to its start point after moving | Expected: `|ΔP|` is straight-line from the first valid sample; distance travelled is path length. |

After any camera, pose, or hands initialization error the program stops the RealSense pipeline, closes MediaPipe, and closes OpenCV windows so the device is not left locked.
