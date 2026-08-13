# Virtual Mouse Control using Hand Gestures

Control your Windows mouse cursor with your index finger, using your webcam
and MediaPipe hand tracking. Pinch your thumb and index finger together to
left-click.

## Files

- `virtual_mouse.py` — complete, ready-to-run source code
- `requirements.txt` — Python dependencies

## 1. Requirements

- **Python 3.9 – 3.11** (MediaPipe's prebuilt wheels are most reliable in this
  range; very new Python versions like 3.13 may not yet have a MediaPipe
  wheel available). Check your version with `python --version`.
- A working webcam
- Windows 10/11 (the code also runs on macOS/Linux, but was optimized/tested
  with Windows in mind, e.g. using the DirectShow camera backend)

## 2. Installation

Open **Command Prompt** or **PowerShell** in the project folder and run:

```bash
# (Recommended) create an isolated virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

If you're on macOS/Linux instead, activate the environment with
`source venv/bin/activate` instead of `venv\Scripts\activate`.

## 3. Running the program

```bash
python virtual_mouse.py
```

A window titled **"Virtual Mouse - Hand Gesture Control"** will open showing
your webcam feed with hand landmarks drawn on top.

- Move your hand so it's visible in the frame, inside the magenta rectangle
  shown on screen (the "active tracking area").
- **Raise your index finger** and move it — the mouse cursor follows.
- **Pinch** your thumb tip and index fingertip together — this fires a single
  left click.
- **Press `q`** (with the webcam window focused) to exit safely. The webcam
  and all windows are released/closed automatically, including on Ctrl+C or
  unexpected errors.

## 4. How it works

### Cursor movement (index finger)
Every frame, MediaPipe Hands returns 21 hand landmarks in normalized (0–1)
coordinates. Landmark `#8` is the index fingertip. The program:

1. Checks whether the index finger is "extended" (fingertip farther from the
   wrist than the middle knuckle of that finger — landmark `#6`).
2. If extended, converts the fingertip's position inside a defined
   **active rectangle** in the webcam frame into full screen coordinates
   (`pyautogui.size()` gives your screen resolution). Using an inset
   rectangle instead of the full frame makes it far easier to reach every
   screen corner without straining your wrist to the edges of the camera view.
3. Passes the mapped coordinates through a **two-stage smoothing filter**
   (a moving average over the last 5 positions, plus exponential smoothing)
   before calling `pyautogui.moveTo(...)`. This removes the small natural
   hand tremor/jitter that would otherwise make the cursor "shake".

### Left click (thumb + index pinch)
1. The Euclidean pixel distance between the thumb tip (`#4`) and index
   fingertip (`#8`) is measured every frame.
2. This distance is **normalized against your hand's own size** (distance
   from wrist to the middle-finger knuckle), so the pinch gesture is
   recognized reliably whether your hand is close to or far from the camera.
3. If the normalized distance drops below a threshold, that frame is flagged
   as "pinching".
4. A **debounce/cooldown state machine** (`ClickDebouncer`) turns this raw
   per-frame signal into one clean click:
   - A click only fires on the transition from *not pinching* → *pinching*
     (not repeatedly while you hold the pinch).
   - A short cooldown (0.6s) blocks a second click from firing too soon.
   - A small number of consecutive "released" frames must be seen before the
     gesture is considered fully released, which prevents jitter around the
     threshold boundary from causing multiple accidental clicks.

## 5. Tuning

All key parameters are constants near the top of `virtual_mouse.py`:

| Constant | Effect |
|---|---|
| `SMOOTHING_WINDOW`, `SMOOTHING_ALPHA` | Higher window / lower alpha = smoother but laggier cursor |
| `PINCH_THRESHOLD_RATIO` | Lower = fingers must get closer together to trigger a click |
| `CLICK_COOLDOWN` | Minimum seconds between clicks |
| `FRAME_MARGIN` | Size of the inset "active tracking rectangle" |
| `CAM_WIDTH`, `CAM_HEIGHT` | Lower resolution = higher FPS, less precision |

## 6. Troubleshooting

- **"Could not open webcam"**: close any other app using the camera (Zoom,
  Teams, etc.), check Windows Settings → Privacy & Security → Camera →
  allow desktop apps to access your camera, and confirm the camera works in
  the Windows Camera app.
- **Cursor feels jittery**: increase `SMOOTHING_WINDOW` or lower
  `SMOOTHING_ALPHA`.
- **Clicks don't register / register too often**: adjust
  `PINCH_THRESHOLD_RATIO` (try values between 0.25 and 0.45) and
  `CLICK_COOLDOWN`.
- **Low FPS**: lower `CAM_WIDTH`/`CAM_HEIGHT` (e.g. 480x360), ensure no other
  heavy app is competing for CPU, and make sure you're not running the
  MediaPipe GPU-less CPU build under unusually heavy system load.
- **`pyautogui.FailSafeException`**: this is intentional — PyAutoGUI aborts
  moves that land exactly in a screen corner as a safety feature. The
  program catches this and simply skips that frame.
