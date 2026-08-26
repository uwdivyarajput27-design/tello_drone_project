"""
enroll_reid.py
Build a mean OSNet re-id embedding for a target person from a video clip,
by averaging embeddings over every YOLO person detection in the clip.

Requirements: opencv-python, torch, torchreid, yolov5 (via torch.hub)
Usage: python enroll_reid.py path/to/video.mp4
"""

import argparse
import pickle

import cv2
import numpy as np
import torch
from torchreid.utils.feature_extractor import FeatureExtractor

OUTPUT_PATH = "me_embedding.pkl"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="Path to a video clip of the target person")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = FeatureExtractor(model_name='osnet_x1_0', device=device)

    model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
    model.conf = 0.4
    model.iou = 0.5
    model.classes = [0]  # person only

    cap = cv2.VideoCapture(args.video)
    all_embeddings = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = model(frame).xyxy[0]
        for det in detections:
            x1, y1, x2, y2, conf, cls = det.tolist()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            features = extractor([person_crop])
            all_embeddings.append(features[0].cpu().numpy())

    cap.release()
    print(f"Total embeddings collected: {len(all_embeddings)}")

    if not all_embeddings:
        print("No person detected in the video.")
        return

    mean_embedding = np.mean(all_embeddings, axis=0)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(mean_embedding, f)

    print(f"Enrollment successful -> {OUTPUT_PATH}")
    print("Embedding shape:", mean_embedding.shape)


if __name__ == "__main__":
    main()
