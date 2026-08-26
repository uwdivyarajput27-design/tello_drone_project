"""
enroll_face.py
Enroll a target face from the webcam for face-based following (see tello_face_follow.py).

Requirements: opencv-python, insightface
Controls: 'c' to capture a sample, 'q' to finish and save.
"""

import cv2
import numpy as np
import pickle
from insightface.app import FaceAnalysis

OUTPUT_PATH = "my_face.pkl"


def main():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))

    cap = cv2.VideoCapture(0)
    embeddings = []

    print("Look at the camera. Press 'c' to capture. Press 'q' to finish.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces = app.get(frame)
            if faces:
                x1, y1, x2, y2 = map(int, faces[0].bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.imshow("Enroll Face", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') and faces:
                embeddings.append(faces[0].embedding)
                print(f"Captured sample {len(embeddings)}")
            elif key == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not embeddings:
        print("No samples captured, nothing saved.")
        return

    mean_embedding = np.mean(embeddings, axis=0)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(mean_embedding, f)

    print(f"Face enrolled successfully -> {OUTPUT_PATH} ({len(embeddings)} samples)")


if __name__ == "__main__":
    main()
