# Models

Model weights are not tracked in this repo (see `.gitignore`). Download the
ones you need into this folder:

- `yolov5n.onnx` — used by `src/person_following/webcam_demo.py` and
  `raspberry_pi_follow.py`. Export from the
  [ultralytics/yolov5](https://github.com/ultralytics/yolov5) repo, or use
  any YOLOv5-format ONNX person detector with a single class-0 (person) head.
- OSNet re-id weights are downloaded automatically by `torchreid` on first
  run of `reid_extractor.py` / `enroll_reid.py`.
- InsightFace's `buffalo_l` model pack is downloaded automatically on first
  run of `enroll_face.py` / `tello_face_follow.py`.
