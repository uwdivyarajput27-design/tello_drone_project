# Tello Code

Hand-gesture and autonomous-following control for the DJI/Ryze Tello drone.

Two independent modules:

- **[Gesture control](src/gesture_control/)** — fly the drone with one-hand
  gestures read from a laptop webcam (MediaPipe Hands).
- **[Person following](src/person_following/)** — enroll a target person by
  face or by full-body re-identification (OSNet), then let the drone (or a
  Raspberry Pi rig) keep them centered in frame automatically.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Some scripts additionally need model weights that are not tracked in this
repo (see [models/README.md](models/README.md)) and, for the drone scripts,
a laptop/Pi connected to the Tello's Wi-Fi access point.

## Modules

| Script | Module | What it does |
|---|---|---|
| `gesture_control.py` | gesture_control | Fly the Tello with hand gestures |
| `enroll_face.py` | person_following | Enroll a face from the webcam |
| `enroll_reid.py` | person_following | Enroll a full-body re-id embedding from a video |
| `webcam_demo.py` | person_following | Highlight the enrolled person on a webcam feed |
| `raspberry_pi_follow.py` | person_following | Lock onto and track the enrolled person from a Pi camera |
| `tello_face_follow.py` | person_following | Autonomous Tello face-following autopilot |

See each module's README for details and usage.

## Safety

The drone scripts (`gesture_control.py`, `tello_face_follow.py`) command
real flight. Always fly with propeller guards, in a large clear space, away
from people and obstacles, and keep a hand on the emergency-land trigger.

## License

MIT — see [LICENSE](LICENSE).
