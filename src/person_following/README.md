# Person Following

Enroll a target person, then have a camera (webcam, Raspberry Pi, or the
Tello itself) keep them locked on and centered.

Two independent recognition paths, pick one:

- **Face-based** (`enroll_face.py` → `tello_face_follow.py`) — uses
  InsightFace face embeddings. Simple, works well when the target's face is
  usually visible to the camera.
- **Full-body re-id** (`enroll_reid.py` → `webcam_demo.py` /
  `raspberry_pi_follow.py`) — uses an OSNet (torchreid) body embedding via
  `reid_extractor.py`, matched against YOLO person detections. Keeps working
  when the target's back is turned or their face is small/occluded.

## 1. Enroll a target

```bash
# Face-based
python enroll_face.py            # press 'c' to sample, 'q' to save -> my_face.pkl

# Re-id-based
python enroll_reid.py path/to/clip.mp4   # -> me_embedding.pkl
```

## 2. Track

```bash
python webcam_demo.py            # webcam re-id demo, highlights the match
python raspberry_pi_follow.py    # Pi camera, search-and-track state machine
python tello_face_follow.py      # full Tello autopilot, face-based (see SAFETY below)
```

## Models

`webcam_demo.py` and `raspberry_pi_follow.py` expect a YOLOv5 ONNX model at
`../../models/yolov5n.onnx` — see [models/README.md](../../models/README.md).
`enroll_reid.py` pulls YOLOv5 via `torch.hub` automatically.

## SAFETY

`tello_face_follow.py` takes off automatically and flies under closed-loop
control. Fly with propeller guards, in a large clear space, and be ready to
take manual control or force-land.
