"""
tello_face_follow.py
Autonomous Tello autopilot: takes off and keeps an enrolled face centered
and at a target distance using proportional control.

Requirements: opencv-python, insightface, numpy, djitellopy
Enrollment: run enroll_face.py first to create my_face.pkl.

SAFETY: this script takes off automatically. Fly in a large, clear,
indoor space with propeller guards fitted, and stand clear of the drone.
"""

import pickle
import time

import cv2
import numpy as np
from djitellopy import Tello
from insightface.app import FaceAnalysis

EMBEDDING_PATH = "my_face.pkl"
SIM_THRESHOLD = 0.45

FRAME_W, FRAME_H = 640, 480
CENTER_X, CENTER_Y = FRAME_W // 2, FRAME_H // 2
DEAD_ZONE = 50            # px of slack around center before correcting

TARGET_AREA = 6000        # face bbox area (px^2) considered "correct distance"
AREA_TOLERANCE = 600

K_FB = 0.004              # forward/back gain (from area error)
K_LR = 0.2                # left/right gain (from x error)
K_UD = 0.2                # up/down gain (from y error)
MAX_SPEED = 25            # clamp on every rc axis


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def clamp_speed(v):
    return int(max(-MAX_SPEED, min(MAX_SPEED, v)))


def tello_get_frame(drone, w=FRAME_W, h=FRAME_H):
    frame = drone.get_frame_read().frame
    return cv2.resize(frame, (w, h))


def main():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))

    with open(EMBEDDING_PATH, "rb") as f:
        target_face = pickle.load(f)

    drone = Tello()
    drone.connect()
    print("Battery:", drone.get_battery())

    drone.streamoff()
    drone.streamon()

    input("Ready to take off. Stand clear and press Enter to continue...")
    drone.takeoff()
    drone.move_up(20)
    time.sleep(4)

    prev_time = 0
    try:
        while True:
            frame = tello_get_frame(drone)
            frame = cv2.flip(frame, 1)

            faces = app.get(frame)
            command_sent = False

            for f in faces:
                sim = cosine_similarity(f.embedding, target_face)
                if sim < SIM_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(int, f.bbox)
                face_cx = (x1 + x2) // 2
                face_cy = (y1 + y2) // 2
                area = (x2 - x1) * (y2 - y1)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Sim: {sim:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                left_right = 0
                up_down = 0
                forward_backward = 0

                if face_cx < CENTER_X - DEAD_ZONE:
                    left_right = clamp_speed(K_LR * (CENTER_X - face_cx))
                elif face_cx > CENTER_X + DEAD_ZONE:
                    left_right = -clamp_speed(K_LR * (face_cx - CENTER_X))

                if face_cy < CENTER_Y - DEAD_ZONE:
                    up_down = clamp_speed(K_UD * (CENTER_Y - face_cy))
                elif face_cy > CENTER_Y + DEAD_ZONE:
                    up_down = -clamp_speed(K_UD * (face_cy - CENTER_Y))

                if area < TARGET_AREA - AREA_TOLERANCE:
                    forward_backward = clamp_speed(K_FB * (TARGET_AREA - area))
                elif area > TARGET_AREA + AREA_TOLERANCE:
                    forward_backward = -clamp_speed(K_FB * (area - TARGET_AREA))

                drone.send_rc_control(left_right, forward_backward, up_down, 0)
                command_sent = True
                print(f"FB: {forward_backward}, LR: {left_right}, UD: {up_down}, "
                      f"pos=({face_cx},{face_cy}), area={area}")
                break  # only follow the first matching face per frame

            if not command_sent:
                drone.send_rc_control(0, 0, 0, 0)

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time else 0
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.imshow("Tello Face Follow", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        drone.send_rc_control(0, 0, 0, 0)
        drone.land()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
