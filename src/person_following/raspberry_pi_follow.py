"""
raspberry_pi_follow.py
Search-and-track a re-id-enrolled target person from a Raspberry Pi camera,
using a lightweight ONNX YOLOv5 model plus IoU-based tracking between detections.

Requirements: opencv-python, onnxruntime, picamera2, numpy
Model: models/yolov5n.onnx (see repo root README for download instructions)
"""

import pickle
import time

import cv2
import numpy as np
import onnxruntime as ort
from numpy.linalg import norm
from picamera2 import Picamera2

from reid_extractor import extract_embedding

# ---------- Config ----------
MODEL_PATH = "../../models/yolov5n.onnx"
EMBEDDING_PATH = "me_embedding.pkl"
FRAME_W, FRAME_H = 360, 240
IMG_SIZE = 320
FRAME_SKIP = 5           # run YOLO every N frames
CONF_THRES = 0.4
SIM_THRES = 0.75
LOCK_TIMEOUT = 2.0       # seconds without a match before the target is dropped
IOU_MATCH_THRES = 0.3


def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


def letterbox(img, new_size=IMG_SIZE, color=(114, 114, 114)):
    h, w, _ = img.shape
    scale = min(new_size / w, new_size / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((new_size, new_size, 3), color, dtype=np.uint8)
    x = (new_size - nw) // 2
    y = (new_size - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def preprocess(img):
    img = img[:, :, ::-1]          # BGR -> RGB
    img = img.transpose(2, 0, 1)   # HWC -> CHW
    img = np.expand_dims(img, 0).astype(np.float32)
    img /= 255.0
    return img


def compute_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def main():
    with open(EMBEDDING_PATH, "rb") as f:
        target_embedding = pickle.load(f)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (FRAME_W, FRAME_H), "format": "RGB888"},
        controls={"FrameRate": 30},
    )
    picam2.configure(config)
    picam2.start()

    target_locked = False
    target_bbox = None
    last_seen_time = 0.0
    frame_id = 0

    start_time = time.time()
    frame_count = 0
    fps = 0

    try:
        while True:
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_id += 1

            if frame_id % FRAME_SKIP == 0:
                img = letterbox(frame, IMG_SIZE)
                inp = preprocess(img)
                preds = session.run([output_name], {input_name: inp})[0][0]
                preds = preds[preds[:, 4] > CONF_THRES]

                detections = []
                for det in preds:
                    if np.argmax(det[5:]) != 0:
                        continue

                    cx, cy, w, h = det[:4]
                    x1 = int((cx - w / 2) * FRAME_W / IMG_SIZE)
                    y1 = int((cy - h / 2) * FRAME_H / IMG_SIZE)
                    x2 = int((cx + w / 2) * FRAME_W / IMG_SIZE)
                    y2 = int((cy + h / 2) * FRAME_H / IMG_SIZE)

                    if (y2 - y1) < 0.25 * FRAME_H:
                        continue

                    detections.append((x1, y1, x2, y2))

                if not target_locked:
                    for bbox in detections:
                        x1, y1, x2, y2 = bbox
                        crop = frame[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue

                        sim = cosine_similarity(extract_embedding(crop), target_embedding)
                        if sim > SIM_THRES:
                            target_locked = True
                            target_bbox = bbox
                            last_seen_time = time.time()
                            print("TARGET LOCKED")
                            break
                else:
                    best_iou, best_bbox = 0.0, None
                    for bbox in detections:
                        iou = compute_iou(bbox, target_bbox)
                        if iou > best_iou:
                            best_iou, best_bbox = iou, bbox

                    if best_bbox and best_iou > IOU_MATCH_THRES:
                        target_bbox = best_bbox
                        last_seen_time = time.time()

            if target_locked and (time.time() - last_seen_time) > LOCK_TIMEOUT:
                print("TARGET LOST")
                target_locked = False
                target_bbox = None

            if target_locked and target_bbox:
                x1, y1, x2, y2 = target_bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "ME", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()

            cv2.putText(frame, f"FPS: {fps:.2f}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Follow Me", frame)
            if cv2.waitKey(1) == 27:
                break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
