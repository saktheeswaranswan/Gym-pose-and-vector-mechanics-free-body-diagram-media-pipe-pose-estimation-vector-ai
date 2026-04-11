# Install required packages
!pip install ultralytics opencv-python numpy -q

import cv2
import numpy as np
import math
import json
from ultralytics import YOLO

# ---------------------------
# CONFIG
# ---------------------------
INPUT_VIDEO = "video.mp4"
OUTPUT_VIDEO = "inference_video.mp4"
OUTPUT_JSON = "timeline.json"
MODEL_PATH = "yolo11n-pose.pt"
MIN_CONF = 0.25

# COCO keypoint indices
KP = {
    "nose": 0, "l_eye": 1, "r_eye": 2, "l_ear": 3, "r_ear": 4,
    "l_sh": 5, "r_sh": 6, "l_el": 7, "r_el": 8, "l_wr": 9, "r_wr": 10,
    "l_hp": 11, "r_hp": 12, "l_kn": 13, "r_kn": 14, "l_an": 15, "r_an": 16
}

# Skeleton connections for COCO-17 pose
SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16)
]

class Vec:
    @staticmethod
    def unit(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else np.zeros_like(v)

    @staticmethod
    def perp(v):
        return np.array([-v[1], v[0]], dtype=np.float32)

    @staticmethod
    def angle(a, b, c):
        ba = Vec.unit(a - b)
        bc = Vec.unit(c - b)
        return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc), -1.0, 1.0))))

class Drawer:
    def point(self, frame, p, color=(0, 255, 255), r=4):
        cv2.circle(frame, tuple(np.round(p).astype(int)), r, color, -1)

    def line(self, frame, p1, p2, color=(200, 200, 200), thickness=2):
        cv2.line(frame, tuple(np.round(p1).astype(int)), tuple(np.round(p2).astype(int)), color, thickness)

    def arrow(self, frame, p, v, color=(255, 0, 0), txt=None):
        end = p + v
        cv2.arrowedLine(
            frame,
            tuple(np.round(p).astype(int)),
            tuple(np.round(end).astype(int)),
            color,
            3,
            tipLength=0.3
        )
        if txt is not None:
            cv2.putText(
                frame,
                txt,
                tuple(np.round(end).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )

    def arc(self, frame, a, b, c, color=(0, 255, 255)):
        ang1 = math.atan2(a[1] - b[1], a[0] - b[0])
        ang2 = math.atan2(c[1] - b[1], c[0] - b[0])
        diff = (ang2 - ang1 + math.pi) % (2 * math.pi) - math.pi

        pts = []
        for t in np.linspace(0, 1, 20):
            ang = ang1 + diff * t
            x = int(b[0] + 30 * math.cos(ang))
            y = int(b[1] + 30 * math.sin(ang))
            pts.append([x, y])

        pts = np.array(pts, np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], False, color, 2)

class Force:
    def compute(self, knee, ankle, hip, com):
        shank = ankle - knee
        normal = Vec.perp(Vec.unit(shank))
        knee_angle = Vec.angle(hip, knee, ankle)
        kf = np.cos(np.radians(knee_angle))

        offset = Vec.unit(com - ankle)
        of = float(np.dot(offset, normal))

        mag = 100 + 120 * abs(kf) + 80 * abs(of)
        vertical = np.array([0, -1], dtype=np.float32)
        direction = Vec.unit(0.6 * normal + 0.4 * vertical)
        return direction * mag, float(mag)

class Stability:
    def compute(self, com, la, ra, width):
        if com is None or (la is None and ra is None):
            return False, 0.0

        feet = [p for p in (la, ra) if p is not None]
        xs = [float(p[0]) for p in feet]
        xmin, xmax = min(xs), max(xs)

        margin = 0.12 * width
        xmin -= margin
        xmax += margin

        dist = 0.0
        if com[0] < xmin:
            dist = xmin - com[0]
        elif com[0] > xmax:
            dist = com[0] - xmax

        score = float(np.exp(-dist / 40.0))
        return (score > 0.5), score

class Engine:
    def __init__(self):
        self.model = YOLO(MODEL_PATH)
        self.draw = Drawer()
        self.force = Force()
        self.stab = Stability()

    def _get(self, kpts, conf, i):
        return kpts[i] if conf[i] > MIN_CONF else None

    def process_person(self, frame, kpts, conf, person_id):
        def get(i):
            return self._get(kpts, conf, i)

        # Skeleton
        for a, b in SKELETON:
            if conf[a] > MIN_CONF and conf[b] > MIN_CONF:
                self.draw.line(frame, kpts[a], kpts[b])

        # Keypoints
        for i in range(len(kpts)):
            if conf[i] > MIN_CONF:
                self.draw.point(frame, kpts[i])

        # Joint angles
        joints = [
            (KP["l_sh"], KP["l_el"], KP["l_wr"]),
            (KP["r_sh"], KP["r_el"], KP["r_wr"]),
            (KP["l_hp"], KP["l_kn"], KP["l_an"]),
            (KP["r_hp"], KP["r_kn"], KP["r_an"])
        ]

        angles = {}
        for a, b, c in joints:
            key = f"{a}_{b}_{c}"
            if conf[a] > MIN_CONF and conf[b] > MIN_CONF and conf[c] > MIN_CONF:
                ang = Vec.angle(kpts[a], kpts[b], kpts[c])
                angles[key] = ang
                self.draw.arc(frame, kpts[a], kpts[b], kpts[c])
                cv2.putText(
                    frame,
                    f"{int(round(ang))}°",
                    tuple(np.round(kpts[b]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )

        # COM from hips and knees if available
        pts = []
        for key in ("l_hp", "r_hp", "l_kn", "r_kn"):
            p = get(KP[key])
            if p is not None:
                pts.append(p)

        com = np.mean(pts, axis=0) if len(pts) else None
        if com is not None:
            cv2.circle(frame, tuple(np.round(com).astype(int)), 6, (255, 0, 255), -1)
            cv2.putText(
                frame,
                f"COM {person_id}",
                tuple(np.round(com).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 255),
                1
            )

        la = get(KP["l_an"])
        ra = get(KP["r_an"])

        left_force = None
        right_force = None

        if get(KP["l_kn"]) is not None and la is not None and get(KP["l_hp"]) is not None and com is not None:
            v, m = self.force.compute(get(KP["l_kn"]), la, get(KP["l_hp"]), com)
            self.draw.arrow(frame, la, v, (255, 0, 0), f"{int(round(m))}")
            left_force = {"vector": [float(v[0]), float(v[1])], "mag": float(m)}

        if get(KP["r_kn"]) is not None and ra is not None and get(KP["r_hp"]) is not None and com is not None:
            v, m = self.force.compute(get(KP["r_kn"]), ra, get(KP["r_hp"]), com)
            self.draw.arrow(frame, ra, v, (255, 0, 0), f"{int(round(m))}")
            right_force = {"vector": [float(v[0]), float(v[1])], "mag": float(m)}

        stable, score = self.stab.compute(com, la, ra, frame.shape[1])

        return {
            "person_id": int(person_id),
            "com": [float(com[0]), float(com[1])] if com is not None else None,
            "left_ankle": [float(la[0]), float(la[1])] if la is not None else None,
            "right_ankle": [float(ra[0]), float(ra[1])] if ra is not None else None,
            "left_force": left_force,
            "right_force": right_force,
            "stability": {"stable": bool(stable), "score": float(score)},
            "angles": angles
        }

    def process_frame(self, frame):
        results = self.model(frame, verbose=False)

        frame_records = []
        if not results or len(results) == 0 or results[0].keypoints is None:
            cv2.putText(
                frame,
                "No person detected",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
            return frame, frame_records

        r = results[0]

        # Multi-person pose:
        # keypoints.xy -> (N, 17, 2)
        # keypoints.conf -> (N, 17)
        kpts_all = r.keypoints.xy.cpu().numpy()
        conf_all = r.keypoints.conf.cpu().numpy()

        if len(kpts_all) == 0:
            cv2.putText(
                frame,
                "No person detected",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
            return frame, frame_records

        # Draw every detected person in the frame
        for pid, (kpts, conf) in enumerate(zip(kpts_all, conf_all)):
            record = self.process_person(frame, kpts, conf, pid)
            frame_records.append(record)

            # Person label at nose if available, otherwise top-left area
            name_xy = None
            if conf[KP["nose"]] > MIN_CONF:
                name_xy = tuple(np.round(kpts[KP["nose"]]).astype(int))
            else:
                valid = kpts[conf > MIN_CONF]
                if len(valid) > 0:
                    p = valid[0]
                    name_xy = tuple(np.round(p).astype(int))
                else:
                    name_xy = (30, 80 + 25 * pid)

            cv2.putText(
                frame,
                f"Person {pid}",
                name_xy,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

        return frame, frame_records

# ---------------------------
# MAIN
# ---------------------------
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open input video: {INPUT_VIDEO}")

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps is None or fps <= 1e-6:
    fps = 30.0

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

engine = Engine()
timeline = []
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    annotated, people = engine.process_frame(frame)
    out.write(annotated)

    timeline.append({
        "frame": frame_idx,
        "time": round(frame_idx / fps, 4),
        "num_persons": len(people),
        "persons": people
    })

    frame_idx += 1

cap.release()
out.release()

with open(OUTPUT_JSON, "w") as f:
    json.dump(timeline, f, indent=2)

print(f"Done. Saved: {OUTPUT_VIDEO}")
print(f"Done. Saved: {OUTPUT_JSON}")
