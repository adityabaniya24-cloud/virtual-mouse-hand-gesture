"""
Virtual Mouse Control using Hand Gestures
==========================================

Controls the computer mouse cursor using your index finger, captured via webcam,
using MediaPipe Hands for real-time hand landmark detection.

Gestures:
- MOVE CURSOR : Raise your index finger (other fingers can be up or down) and
                move your hand. The index fingertip position is mapped from the
                webcam frame to your screen resolution to move the cursor.
- LEFT CLICK  : Bring your thumb tip and index fingertip close together (a
                "pinch" gesture). A click fires once per pinch, with a cooldown
                to prevent repeated/accidental clicks.
- QUIT        : Press 'q' with the webcam window focused.

Tested with:
    Python 3.9 - 3.11 (MediaPipe currently does not support 3.13 on all
    platforms; 3.9-3.11 is the safest range as of 2025/2026)
    mediapipe >= 0.10 (new solutions API used: mediapipe.solutions.hands)
    opencv-python >= 4.8
    pyautogui >= 0.9.54
    numpy >= 1.24

Author: Generated for Windows desktop use, but works cross-platform.
"""

import sys
import time
import math
from collections import deque

import cv2
import numpy as np
import pyautogui

try:
    import mediapipe as mp
except ImportError:
    print("ERROR: mediapipe is not installed. Run: pip install -r requirements.txt")
    sys.exit(1)


# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------

# Webcam capture resolution (smaller = faster processing = better real-time FPS)
CAM_WIDTH, CAM_HEIGHT = 640, 480

# Webcam device index. 0 is the default/built-in camera on most Windows PCs.
CAM_INDEX = 0

# Margin (in pixels, within the webcam frame) that defines the "active
# tracking rectangle". Keeping the finger away from the extreme edges of the
# frame makes it much easier to reach the corners of the screen, since MediaPipe
# tracking quality drops near frame edges and your hand naturally can't reach
# the physical edge of the camera view.
FRAME_MARGIN = 100

# Number of previous cursor positions to average over for smoothing.
# Higher = smoother but more "lag". Lower = more responsive but shakier.
SMOOTHING_WINDOW = 5

# Extra exponential smoothing factor applied on top of the moving average.
# 0 < ALPHA <= 1. Lower ALPHA = smoother/slower, higher ALPHA = snappier.
SMOOTHING_ALPHA = 0.4

# Distance (in normalized pixel space of the webcam frame) between thumb tip
# and index fingertip below which we consider it a "pinch" (click gesture).
# This is auto-scaled relative to the detected hand size (see get_pinch_threshold).
PINCH_THRESHOLD_RATIO = 0.35  # relative to a reference hand-size distance

# Minimum time (seconds) between two consecutive clicks - prevents rapid-fire
# / accidental repeated clicks while the pinch gesture is held.
CLICK_COOLDOWN = 0.6

# Minimum time the fingers must be released (un-pinched) before another click
# can register. This debounces jitter around the pinch threshold boundary.
RELEASE_CONFIRM_FRAMES = 2

# MediaPipe Hands detection/tracking confidence
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7

# Landmark indices (from MediaPipe Hands topology)
THUMB_TIP = 4
INDEX_FINGER_TIP = 8
INDEX_FINGER_PIP = 6
MIDDLE_FINGER_TIP = 12
MIDDLE_FINGER_PIP = 10
RING_FINGER_TIP = 16
RING_FINGER_PIP = 14
PINKY_TIP = 20
PINKY_PIP = 18
WRIST = 0


# ----------------------------------------------------------------------------
# HELPER CLASSES
# ----------------------------------------------------------------------------

class CursorSmoother:
    """
    Smooths noisy fingertip coordinates before moving the OS cursor.

    Combines two techniques:
      1. A moving average over the last N positions (reduces high-frequency jitter).
      2. Exponential smoothing (weighted blend of new vs. previous smoothed value)
         to further reduce shake while staying responsive to intentional movement.
    """

    def __init__(self, window_size=SMOOTHING_WINDOW, alpha=SMOOTHING_ALPHA):
        self.window = deque(maxlen=window_size)
        self.alpha = alpha
        self.prev_x = None
        self.prev_y = None

    def update(self, x, y):
        # Step 1: moving average
        self.window.append((x, y))
        avg_x = sum(p[0] for p in self.window) / len(self.window)
        avg_y = sum(p[1] for p in self.window) / len(self.window)

        # Step 2: exponential smoothing against the previous output
        if self.prev_x is None:
            smooth_x, smooth_y = avg_x, avg_y
        else:
            smooth_x = self.prev_x + self.alpha * (avg_x - self.prev_x)
            smooth_y = self.prev_y + self.alpha * (avg_y - self.prev_y)

        self.prev_x, self.prev_y = smooth_x, smooth_y
        return smooth_x, smooth_y

    def reset(self):
        self.window.clear()
        self.prev_x = None
        self.prev_y = None


class ClickDebouncer:
    """
    Turns a raw, noisy "is pinching?" boolean signal into clean, single-shot
    click events, with a cooldown and a release-confirmation step so that:
      - A click fires only once per pinch gesture (not repeatedly while held).
      - Fast jitter around the pinch threshold does not cause multiple clicks.
      - A minimum time must pass between clicks even across separate pinches.
    """

    def __init__(self, cooldown=CLICK_COOLDOWN, release_frames=RELEASE_CONFIRM_FRAMES):
        self.cooldown = cooldown
        self.release_frames = release_frames
        self.is_pinching = False       # current debounced pinch state
        self.release_counter = 0       # consecutive "not pinching" frames seen
        self.last_click_time = 0.0

    def update(self, pinch_detected_now: bool):
        """
        Call once per frame with the raw pinch detection result.
        Returns True exactly on the frame a click should be fired.
        """
        fire_click = False
        now = time.time()

        if pinch_detected_now:
            self.release_counter = 0
            if not self.is_pinching:
                # Transition: released -> pinched => candidate click
                if (now - self.last_click_time) >= self.cooldown:
                    fire_click = True
                    self.last_click_time = now
                self.is_pinching = True
        else:
            self.release_counter += 1
            if self.release_counter >= self.release_frames:
                self.is_pinching = False

        return fire_click


# ----------------------------------------------------------------------------
# GEOMETRY HELPERS
# ----------------------------------------------------------------------------

def euclidean_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def is_finger_extended(landmarks, tip_idx, pip_idx, wrist_idx=WRIST):
    """
    Simple heuristic: a finger is considered "extended" if its tip is farther
    from the wrist than its PIP joint is. Works regardless of hand rotation
    reasonably well for the index/middle/ring/pinky fingers.
    """
    tip = landmarks[tip_idx]
    pip = landmarks[pip_idx]
    wrist = landmarks[wrist_idx]
    return euclidean_distance(tip, wrist) > euclidean_distance(pip, wrist)


def map_range(value, in_min, in_max, out_min, out_max):
    """Linearly map value from [in_min, in_max] to [out_min, out_max], with clamping."""
    value = max(min(value, in_max), in_min)
    return out_min + (float(value - in_min) / float(in_max - in_min)) * (out_max - out_min)


# ----------------------------------------------------------------------------
# MAIN APPLICATION
# ----------------------------------------------------------------------------

def main():
    # --- PyAutoGUI safety / behaviour settings ---
    # Disable the built-in "fail-safe" corner-abort ONLY if you find it too
    # sensitive; leaving it True (default) means moving the cursor to a screen
    # corner triggers a safe abort exception. We keep it enabled for safety.
    pyautogui.FAILSAFE = True
    # Removes the small artificial delay PyAutoGUI adds after each call,
    # which is essential for smooth, real-time cursor movement.
    pyautogui.PAUSE = 0

    screen_width, screen_height = pyautogui.size()
    print(f"Detected screen resolution: {screen_width}x{screen_height}")

    # --- Initialize webcam ---
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)  # CAP_DSHOW = fast/reliable backend on Windows
    if not cap.isOpened():
        # Fallback: try without the DirectShow backend hint (e.g. non-Windows OS)
        cap = cv2.VideoCapture(CAM_INDEX)

    if not cap.isOpened():
        print("ERROR: Could not open webcam. Please check that:")
        print("  1. A webcam is connected.")
        print("  2. No other application is currently using the webcam.")
        print("  3. Windows camera privacy settings allow desktop apps to access the camera.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    # --- Initialize MediaPipe Hands ---
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,      # video stream mode (uses tracking between frames -> faster)
        max_num_hands=1,               # we only need one hand for mouse control
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )

    smoother = CursorSmoother()
    click_debouncer = ClickDebouncer()

    # FPS calculation
    prev_frame_time = 0.0

    print("Virtual Mouse started.")
    print(" - Raise your index finger and move it to control the cursor.")
    print(" - Pinch thumb + index finger together to left-click.")
    print(" - Press 'q' in the webcam window to quit.")

    try:
        while True:
            success, frame = cap.read()
            if not success or frame is None:
                print("WARNING: Failed to read frame from webcam. Retrying...")
                time.sleep(0.05)
                continue

            # Mirror the frame horizontally so movement feels natural (like a mirror),
            # matching how the user intuitively expects left/right to map.
            frame = cv2.flip(frame, 1)
            frame_h, frame_w, _ = frame.shape

            # MediaPipe expects RGB input; OpenCV captures in BGR.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False  # perf optimization: avoid unnecessary copy
            results = hands.process(rgb_frame)
            rgb_frame.flags.writeable = True

            # Draw the "active tracking rectangle" for user reference
            cv2.rectangle(
                frame,
                (FRAME_MARGIN, FRAME_MARGIN),
                (frame_w - FRAME_MARGIN, frame_h - FRAME_MARGIN),
                (255, 0, 255),
                2,
            )

            status_text = "No hand detected"
            status_color = (0, 0, 255)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                # Draw the hand skeleton on the frame for visual feedback
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                # Convert normalized landmarks (0.0-1.0) to pixel coordinates
                landmarks_px = [
                    (lm.x * frame_w, lm.y * frame_h) for lm in hand_landmarks.landmark
                ]

                index_tip = landmarks_px[INDEX_FINGER_TIP]
                thumb_tip = landmarks_px[THUMB_TIP]

                index_up = is_finger_extended(landmarks_px, INDEX_FINGER_TIP, INDEX_FINGER_PIP)

                # --- CURSOR MOVEMENT ---
                if index_up:
                    status_text = "Tracking: index finger up"
                    status_color = (0, 255, 0)

                    # Map index fingertip position from the "active rectangle"
                    # inside the webcam frame to full screen coordinates.
                    screen_x = map_range(
                        index_tip[0], FRAME_MARGIN, frame_w - FRAME_MARGIN, 0, screen_width
                    )
                    screen_y = map_range(
                        index_tip[1], FRAME_MARGIN, frame_h - FRAME_MARGIN, 0, screen_height
                    )

                    smooth_x, smooth_y = smoother.update(screen_x, screen_y)

                    # pyautogui.moveTo with duration=0 gives an immediate, non-blocking
                    # move -- essential to keep up with real-time frame rate.
                    try:
                        pyautogui.moveTo(smooth_x, smooth_y, duration=0)
                    except pyautogui.FailSafeException:
                        # Cursor hit a screen corner (fail-safe trigger) - ignore this frame.
                        pass

                    cv2.circle(frame, (int(index_tip[0]), int(index_tip[1])), 10, (0, 255, 0), cv2.FILLED)
                else:
                    status_text = "Index finger down (cursor paused)"
                    status_color = (0, 165, 255)
                    smoother.reset()  # avoid a jump when tracking resumes

                # --- CLICK DETECTION (thumb-index pinch) ---
                # Scale the pinch threshold to the hand's own size (distance from
                # wrist to middle-finger MCP acts as a stable reference), so the
                # click gesture works consistently whether your hand is close to
                # or far from the camera.
                hand_size_ref = euclidean_distance(landmarks_px[WRIST], landmarks_px[9])  # 9 = middle finger MCP
                hand_size_ref = max(hand_size_ref, 1e-6)  # avoid div-by-zero
                pinch_distance = euclidean_distance(thumb_tip, index_tip)
                pinch_ratio = pinch_distance / hand_size_ref

                is_pinching_now = pinch_ratio < PINCH_THRESHOLD_RATIO

                # Visual feedback line between thumb and index finger
                line_color = (0, 255, 255) if is_pinching_now else (255, 255, 0)
                cv2.line(
                    frame,
                    (int(thumb_tip[0]), int(thumb_tip[1])),
                    (int(index_tip[0]), int(index_tip[1])),
                    line_color,
                    2,
                )

                if click_debouncer.update(is_pinching_now):
                    pyautogui.click(button="left")
                    status_text = "LEFT CLICK!"
                    status_color = (0, 0, 255)
                    # Visual flash feedback on click
                    cv2.circle(frame, (int(index_tip[0]), int(index_tip[1])), 20, (0, 0, 255), cv2.FILLED)
            else:
                smoother.reset()

            # --- FPS overlay ---
            curr_frame_time = time.time()
            fps = 1.0 / (curr_frame_time - prev_frame_time) if prev_frame_time else 0.0
            prev_frame_time = curr_frame_time

            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2)
            cv2.putText(frame, status_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, status_color, 2)
            cv2.putText(frame, "Press 'q' to quit", (10, frame_h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("Virtual Mouse - Hand Gesture Control", frame)

            # Poll for the quit key. waitKey(1) keeps the loop responsive for
            # real-time video while still checking for keypresses every frame.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quit key pressed. Exiting...")
                break

    except KeyboardInterrupt:
        print("Interrupted by user (Ctrl+C). Exiting...")

    finally:
        # Always release resources cleanly, even if an error occurred.
        hands.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam released and windows closed. Goodbye!")


if __name__ == "__main__":
    main()
