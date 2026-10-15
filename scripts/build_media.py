"""Generate local previews from the supplied recording and COCO skeletons."""
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "static" / "media"
OUT.mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(str(ROOT / "samples" / "fall-01-cam0.mp4"))
try:
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Original preview video is unreadable")
    cv2.imwrite(str(OUT / "room-preview.jpg"), frame)
finally:
    cap.release()

edges = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
for exercise in ("chair_sit_to_stand", "squat", "standing_leg_raise", "hurdle_step"):
    record = json.loads((ROOT / "algorithm2" / "examples" / f"{exercise}_replay.json").read_text())
    poses = [np.array(f["keypoints"]) for f in record["frames"][:250]]
    # Choose a visibly non-neutral recorded posture, for a useful activity preview.
    score = lambda p: abs(p[13, 1] - p[14, 1]) if exercise in {"standing_leg_raise", "hurdle_step"} else abs(p[11, 0] - p[13, 0])
    pose = max(poses, key=score)
    valid = pose[pose[:, 2] >= .3, :2]
    lo, hi = valid.min(0), valid.max(0)
    scale = min(250 / max(hi[0]-lo[0], 1), 210 / max(hi[1]-lo[1], 1))
    points = (pose[:, :2] - (lo+hi)/2) * scale + [200, 130]
    image = np.full((260, 400, 4), 0, dtype=np.uint8)
    for a, b in edges:
        if pose[a, 2] < .3 or pose[b, 2] < .3:
            continue
        color = (111, 151, 74, 255) if a % 2 else (177, 127, 74, 255)
        cv2.line(image, tuple(points[a].astype(int)), tuple(points[b].astype(int)), color, 8, cv2.LINE_AA)
    for i in range(5, 17):
        if pose[i, 2] >= .3:
            pt = tuple(points[i].astype(int))
            cv2.circle(image, pt, 6, (127, 157, 119, 255), -1, cv2.LINE_AA)
            cv2.circle(image, pt, 3, (246, 250, 247, 255), -1, cv2.LINE_AA)
    shoulders = (points[5] + points[6]) / 2
    hips = (points[11] + points[12]) / 2
    head = shoulders + (shoulders - hips) * .31
    cv2.circle(image, tuple(head.astype(int)), 13, (134, 162, 122, 255), 5, cv2.LINE_AA)
    cv2.imwrite(str(OUT / f"{exercise}.png"), image)
print("Built preview images from original data")
