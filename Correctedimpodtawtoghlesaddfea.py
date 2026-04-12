# ---------------------------
# INSTALL (run once in Colab)
# ---------------------------
# !pip install ultralytics opencv-python numpy -q

# ---------------------------
# IMPORTS
# ---------------------------
import cv2
import csv
import math
import numpy as np
from collections import deque
from ultralytics import YOLO

# ---------------------------
# CONFIG
# ---------------------------
INPUT_VIDEO = "video.mp4"
OUTPUT_VIDEO = "inference_video.mp4"
OUTPUT_CSV = "timeline.csv"
MODEL_PATH = "yolo11n-pose.pt"

MIN_CONF = 0.25
SHOW_LIVE = False

# ---------------------------
# BIOMECHANICS PARAMETERS
# ---------------------------
MASS_KG = 70.0
GRAVITY = 9.81
MU_FRICTION = 0.65
HISTORY = 10

# ---------------------------
# ✅ DRAW TOGGLES (CORRECT PLACE)
# ---------------------------
DRAW_SUPPORTHULL = True
DRAW_NORMALS = True
DRAW_COM_COG = True
DRAW_REACTION_VECTORS = True
DRAW_JOINT_ANGLES = True
DRAW_PERSON_LABEL = True
DRAW_FRICTION_CONE = True
DRAW_PRESSURE_CENTER = True
DRAW_GROUND_REACTION_FORCES = True
DRAW_BODY_VECTOR = True
DRAW_FALL_RISK = True
DRAW_KEYPOINT_LABELS = False

# ---------------------------
# KEYPOINTS
# ---------------------------
KP = {
    "nose": 0, "l_eye": 1, "r_eye": 2, "l_ear": 3, "r_ear": 4,
    "l_sh": 5, "r_sh": 6, "l_el": 7, "r_el": 8,
    "l_wr": 9, "r_wr": 10, "l_hp": 11, "r_hp": 12,
    "l_kn": 13, "r_kn": 14, "l_an": 15, "r_an": 16
}

SKELETON = [
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16)
]

JOINTS = {
    "left_elbow": (5,7,9),
    "right_elbow": (6,8,10),
    "left_knee": (11,13,15),
    "right_knee": (12,14,16)
}

# ---------------------------
# VECTOR UTILS
# ---------------------------
class Vec:

    @staticmethod
    def unit(v):
        n = np.linalg.norm(v)
        return v/n if n > 1e-6 else np.zeros_like(v)

    @staticmethod
    def angle(a, b, c):
        ba = Vec.unit(a - b)
        bc = Vec.unit(c - b)
        return np.degrees(np.arccos(np.clip(np.dot(ba, bc), -1, 1)))

# ---------------------------
# DRAW CLASS (FIXED)
# ---------------------------
class Draw:

    @staticmethod
    def point(img, p, c=(0,255,255), r=5):
        if p is None: return
        cv2.circle(img, (int(p[0]), int(p[1])), r, c, -1)

    @staticmethod
    def line(img, a, b, c=(200,200,200), t=2):
        if a is None or b is None: return
        cv2.line(img, (int(a[0]),int(a[1])), (int(b[0]),int(b[1])), c, t)

    @staticmethod
    def text(img, t, p, c=(255,255,255), s=0.5):
        if p is None: return
        cv2.putText(img, t, (int(p[0]),int(p[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, s, c, 1)

    # ✅ FIXED ARC FUNCTION
    @staticmethod
    def arc(img, a, b, c, color=(0,255,255), radius=25):

        a = np.array(a); b = np.array(b); c = np.array(c)

        ba = a - b
        bc = c - b

        if np.linalg.norm(ba) < 1e-6 or np.linalg.norm(bc) < 1e-6:
            return

        ang1 = math.atan2(ba[1], ba[0])
        ang2 = math.atan2(bc[1], bc[0])

        if ang2 < ang1:
            ang1, ang2 = ang2, ang1

        for t in np.linspace(ang1, ang2, 20):
            x = int(b[0] + radius * math.cos(t))
            y = int(b[1] + radius * math.sin(t))
            cv2.circle(img, (x,y), 1, color, -1)

# ---------------------------
# ENGINE
# ---------------------------
class Engine:

    def __init__(self):
        self.model = YOLO(MODEL_PATH)
        self.draw = Draw()

    def process_person(self, frame, kpts, conf):

        # draw skeleton
        for a,b in SKELETON:
            if conf[a] > MIN_CONF and conf[b] > MIN_CONF:
                self.draw.line(frame, kpts[a], kpts[b])

        # draw joints
        for name,(a,b,c) in JOINTS.items():
            if conf[a]>MIN_CONF and conf[b]>MIN_CONF and conf[c]>MIN_CONF:

                pa, pb, pc = kpts[a], kpts[b], kpts[c]
                ang = Vec.angle(pa,pb,pc)

                if DRAW_JOINT_ANGLES:
                    self.draw.arc(frame, pa,pb,pc)
                    self.draw.text(frame, f"{int(ang)}°",
                                   (pb[0]+5, pb[1]-5))

        return frame

    def process_frame(self, frame):

        results = self.model(frame, verbose=False)

        if not results or results[0].keypoints is None:
            return frame

        kpts_all = results[0].keypoints.xy.cpu().numpy()
        conf_all = results[0].keypoints.conf.cpu().numpy()

        for kpts, conf in zip(kpts_all, conf_all):
            frame = self.process_person(frame, kpts, conf)

        return frame

# ---------------------------
# MAIN
# ---------------------------
def main():

    cap = cv2.VideoCapture(INPUT_VIDEO)
    w = int(cap.get(3))
    h = int(cap.get(4))
    fps = cap.get(5) or 30

    out = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (w,h)
    )

    engine = Engine()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = engine.process_frame(frame)
        out.write(frame)

        if SHOW_LIVE:
            cv2.imshow("Output", frame)
            if cv2.waitKey(1) == 27:
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Done")

# ---------------------------
# ENTRY POINT
# ---------------------------
if __name__ == "__main__":
    main()
