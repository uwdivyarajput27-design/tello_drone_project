"""
gesture_control.py
Control a DJI/Ryze Tello drone with one-hand gestures read from a webcam.

Requirements: opencv-python, mediapipe, djitellopy
Run: connect your laptop to the Tello's Wi-Fi, then `python gesture_control.py`
"""

import time
import cv2
import mediapipe as mp
from djitellopy import Tello
from collections import deque

# ---------- Config ----------
STABLE_FRAMES = 6      # frames of history used to debounce jitter
COOLDOWN = 1.2          # seconds between triggers of the same action
SHOW_WEBCAM = True      # set False to hide the camera preview window

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def get_finger_pattern(hand_landmarks):
    """
    Returns a binary tuple: (Thumb, Index, Middle, Ring, Pinky)
    1 = finger extended, 0 = finger curled.
    Assumes a mirrored (selfie-view) frame.
    """
    lm = hand_landmarks.landmark

    thumb_open = 1 if lm[4].x < lm[3].x else 0

    finger_tips = [8, 12, 16, 20]
    fingers_open = [1 if lm[tip].y < lm[tip - 2].y else 0 for tip in finger_tips]

    return (thumb_open, *fingers_open)


def build_gesture_mapping(tello: Tello, state: dict):
    """
    Maps a (Thumb, Index, Middle, Ring, Pinky) pattern to an action.
    `state` carries mutable flight state (currently just `flying`) between calls.
    """

    def takeoff_or_up():
        if state["flying"]:
            tello.send_rc_control(0, 0, 30, 0)
        else:
            tello.takeoff()
            state["flying"] = True

    def land():
        if state["flying"]:
            tello.send_rc_control(0, 0, 0, 0)
            tello.land()
            state["flying"] = False

    return {
        (1, 1, 1, 1, 1): ("Takeoff / Up", takeoff_or_up),
        (0, 1, 1, 1, 0): ("Land", land),
        (0, 1, 1, 1, 1): ("Down", lambda: tello.send_rc_control(0, 0, -30, 0)),
        (0, 1, 0, 0, 0): ("Forward", lambda: tello.send_rc_control(0, 70, 0, 0)),
        (0, 1, 1, 0, 0): ("Backward", lambda: tello.send_rc_control(0, -70, 0, 0)),
        (1, 0, 0, 0, 0): ("Left", lambda: tello.send_rc_control(-70, 0, 0, 0)),
        (0, 0, 0, 0, 1): ("Right", lambda: tello.send_rc_control(70, 0, 0, 0)),
        (0, 0, 0, 0, 0): ("Hover", lambda: tello.send_rc_control(0, 0, 0, 0)),
    }


def main():
    print("[INFO] Connecting to Tello...")
    tello = Tello()
    try:
        tello.connect()
    except Exception as e:
        print("Failed to connect to Tello. Make sure you're on its Wi-Fi. Error:", e)
        return

    print("[INFO] Battery:", tello.get_battery())

    flight_state = {"flying": False}
    mapping = build_gesture_mapping(tello, flight_state)

    hands = mp_hands.Hands(
        min_detection_confidence=0.65, min_tracking_confidence=0.6, max_num_hands=1
    )

    cap = cv2.VideoCapture(0)
    pattern_history = deque(maxlen=STABLE_FRAMES)
    last_action_name = None
    last_trigger_time = 0

    print("[INFO] Starting webcam. Show gestures to the camera.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera not available")
                break

            frame = cv2.flip(frame, 1)  # mirror for selfie view
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            if res.multi_hand_landmarks:
                hand = res.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                pattern = get_finger_pattern(hand)
                pattern_history.append(pattern)

                if len(pattern_history) == pattern_history.maxlen:
                    stable_pattern = max(set(pattern_history), key=pattern_history.count)
                    now = time.time()
                    if (
                        stable_pattern == pattern
                        and pattern in mapping
                        and (now - last_trigger_time) > COOLDOWN
                    ):
                        action_name, action_func = mapping[pattern]
                        print(f"[TRIGGER] {action_name} for pattern {pattern}")
                        try:
                            action_func()
                        except Exception as e:
                            print("Action function error:", e)
                        last_action_name = action_name
                        last_trigger_time = now

                cv2.putText(
                    frame, f"Pattern: {pattern}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                )
            else:
                pattern_history.clear()

            if last_action_name and (time.time() - last_trigger_time) < 3:
                cv2.putText(
                    frame, f"Last: {last_action_name}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2,
                )

            if SHOW_WEBCAM:
                cv2.imshow("Tello Gesture Control", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    print("Quitting due to key press")
                    break
            else:
                time.sleep(0.01)

    finally:
        print("[INFO] Cleaning up")
        try:
            tello.send_rc_control(0, 0, 0, 0)
            if flight_state["flying"]:
                tello.land()
        except Exception:
            pass
        cap.release()
        cv2.destroyAllWindows()
        try:
            tello.end()
        except Exception:
            pass


if __name__ == "__main__":
    main()
