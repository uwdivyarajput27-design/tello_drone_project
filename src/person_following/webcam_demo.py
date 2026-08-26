"""
webcam_demo.py
Live webcam demo: detect people with a YOLOv5 ONNX model, and outline the
one whose OSNet re-id embedding matches the enrolled target (see enroll_reid.py).

Requirements: opencv-python, onnxruntime, numpy
Model: models/yolov5n.onnx (see repo root README for download instructions)
"""

import pickle
import time

import cv2
import numpy as np
import onnxruntime as ort
from numpy.linalg import norm

from reid_extractor import extract_embedding

MODEL_PATH = "../../models/yolov5n.onnx"
EMBEDDING_PATH = "me_embedding.pkl"
CONF_THRES = 0.4
SIM_THRES = 0.8
IMG_SIZE = 640


def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


def preprocess(frame, img_size=IMG_SIZE):
    img = cv2.resize(frame, (img_size, img_size))
    img = img[:, :, ::-1]  # BGR -> RGB
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0).astype(np.float32)
    img /= 255.0
    return img


def main():
    with open(EMBEDDING_PATH, "rb") as f:
        target_embedding = pickle.load(f)

    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    start_time = time.time()
    frame_count = 0
    fps = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            input_tensor = preprocess(frame)
            outputs = session.run([output_name], {input_name: input_tensor})
            preds = outputs[0][0]
            preds = preds[preds[:, 4] > CONF_THRES]

            for det in preds:
                conf = det[4]
                cls = np.argmax(det[5:])
                if cls != 0 or conf <= CONF_THRES:
                    continue

                cx, cy, w, h = det[:4]
                x1 = int((cx - w / 2) * frame.shape[1] / IMG_SIZE)
                y1 = int((cy - h / 2) * frame.shape[0] / IMG_SIZE)
                x2 = int((cx + w / 2) * frame.shape[1] / IMG_SIZE)
                y2 = int((cy + h / 2) * frame.shape[0] / IMG_SIZE)

                person_crop = frame[y1:y2, x1:x2]
                if person_crop.size == 0:
                    continue

                embedding = extract_embedding(person_crop)
                similarity = cosine_similarity(embedding, target_embedding)
                if similarity > SIM_THRES:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()

            cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Person Re-ID Demo", frame)
            if cv2.waitKey(1) == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
