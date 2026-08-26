# Gesture Control

Fly a Tello drone using one-hand gestures captured by a webcam
(MediaPipe Hands for landmark detection, djitellopy for the drone link).

## Run

1. Connect your laptop to the Tello's Wi-Fi access point.
2. `python gesture_control.py`

## Gestures

| Pattern (Thumb, Index, Middle, Ring, Pinky) | Action |
|---|---|
| ✋ all open | Takeoff, then Up on repeat |
| 🖖 index+middle+ring | Land |
| ✊ index+middle+ring+pinky | Down |
| ☝️ index only | Forward |
| ✌️ index+middle | Backward |
| 👍 thumb only | Left |
| 🤙 pinky only | Right |
| ✊ fist | Hover |

Edit `build_gesture_mapping()` in `gesture_control.py` to change the mapping.

## Config

- `STABLE_FRAMES` — how many consecutive frames must agree before a gesture
  is accepted (debounces jitter).
- `COOLDOWN` — minimum seconds between two triggers.
- `SHOW_WEBCAM` — set `False` to run headless.
