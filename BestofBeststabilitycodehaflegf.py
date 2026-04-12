# ---------------------------
# INSTALL
# ---------------------------
# !pip install ultralytics opencv-python numpy -q

import cv2
import numpy as np
import math
from ultralytics import YOLO
from collections import deque

# ---------------------------
# CONFIG
# ---------------------------
INPUT_VIDEO = "video.mp4"
OUTPUT_VIDEO = "inference_video.mp4"
MODEL_PATH = "yolo11n-pose.pt"

MIN_CONF = 0.3
HISTORY = 10

# ---------------------------
# KEYPOINTS
# ---------------------------
KP = {
    "l_sh": 5, "r_sh": 6,
    "l_el": 7, "r_el": 8,
    "l_wr": 9, "r_wr": 10,
    "l_hp": 11, "r_hp": 12,
    "l_kn": 13, "r_kn": 14,
    "l_an": 15, "r_an": 16
}

# ---------------------------
# VECTOR UTIL
# ---------------------------
class Vec:
    @staticmethod
    def unit(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else np.zeros_like(v)

    @staticmethod
    def dot(a, b):
        return float(np.dot(a, b))

    @staticmethod
    def cross2d(a, b):
        return float(a[0]*b[1] - a[1]*b[0])

# ---------------------------
# BIOMECHANICS
# ---------------------------
class Biomech:

    @staticmethod
    def compute_com(kpts, conf):
        pts = [np.array(p) for i,p in enumerate(kpts) if conf[i]>MIN_CONF]
        return np.mean(pts, axis=0) if len(pts)>0 else None

    @staticmethod
    def compute_cop(l_an, r_an, com):
        if l_an is None and r_an is None:
            return None
        if l_an is None:
            return r_an
        if r_an is None:
            return l_an
        if com is None:
            return (l_an + r_an) / 2

        dl = abs(com[0] - l_an[0])
        dr = abs(com[0] - r_an[0])

        wl = 1 / (dl + 1e-6)
        wr = 1 / (dr + 1e-6)

        s = wl + wr
        wl /= s
        wr /= s

        return wl * l_an + wr * r_an

    @staticmethod
    def stability(com, cop):
        if com is None or cop is None:
            return False, 0.0

        dx = abs(com[0] - cop[0])
        score = math.exp(-dx / 50)
        return score > 0.55, score

    @staticmethod
    def joint_torque(joint, com, force):
        if joint is None or com is None:
            return 0.0
        r = com - joint
        return Vec.cross2d(r, force)

    @staticmethod
    def leg_normal_force(hip, ankle, com):
        if hip is None or ankle is None or com is None:
            return np.zeros(2, dtype=np.float32)

        leg = Vec.unit(ankle - hip)
        gravity = np.array([0, 1], dtype=np.float32)

        load = Vec.dot(gravity, leg)
        return leg * load * 150

    @staticmethod
    def hand_force(sh, el, wr):
        if sh is None or el is None or wr is None:
            return np.zeros(2, dtype=np.float32)

        arm = Vec.unit(wr - sh)
        return arm * 80


# ---------------------------
# DRAW
# ---------------------------
class Draw:

    @staticmethod
    def point(img, p, c, r=5):
        cv2.circle(img, (int(p[0]), int(p[1])), r, c, -1)

    @staticmethod
    def arrow(img, p, v, scale=0.4, c=(0,0,255)):
        end = p + v * scale
        cv2.arrowedLine(img,
                        (int(p[0]), int(p[1])),
                        (int(end[0]), int(end[1])),
                        c, 2)

    @staticmethod
    def text(img, t, p, c=(255,255,255)):
        cv2.putText(img, t,
                    (int(p[0]), int(p[1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, c, 1)


# ---------------------------
# ENGINE
# ---------------------------
class Engine:

    def __init__(self):
        self.model = YOLO(MODEL_PATH)
        self.history = {}

    def predict_fall(self, pid, com):
        if com is None:
            return False

        if pid not in self.history:
            self.history[pid] = deque(maxlen=HISTORY)

        hist = self.history[pid]
        hist.append(com.copy())

        if len(hist) < HISTORY:
            return False

        velocities = [hist[i+1][0] - hist[i][0] for i in range(len(hist)-1)]
        trend = np.mean(velocities)

        return abs(trend) > 5

    def process(self, frame):

        results = self.model(frame, verbose=False)
        if results[0].keypoints is None:
            return frame

        kpts_all = results[0].keypoints.xy.cpu().numpy()
        conf_all = results[0].keypoints.conf.cpu().numpy()

        for pid, (kpts, conf) in enumerate(zip(kpts_all, conf_all)):

            pts = {k: np.array(kpts[i]) if conf[i]>MIN_CONF else None
                   for k,i in KP.items()}

            com = Biomech.compute_com(kpts, conf)
            cop = Biomech.compute_cop(pts["l_an"], pts["r_an"], com)

            stable, score = Biomech.stability(com, cop)
            fall = self.predict_fall(pid, com)

            grf = np.array([0, -150], dtype=np.float32)

            torque_l = Biomech.joint_torque(pts["l_kn"], com, grf)
            torque_r = Biomech.joint_torque(pts["r_kn"], com, grf)

            nf_l = Biomech.leg_normal_force(pts["l_hp"], pts["l_an"], com)
            nf_r = Biomech.leg_normal_force(pts["r_hp"], pts["r_an"], com)

            hf_l = Biomech.hand_force(pts["l_sh"], pts["l_el"], pts["l_wr"])
            hf_r = Biomech.hand_force(pts["r_sh"], pts["r_el"], pts["r_wr"])

            # DRAW
            if com is not None:
                Draw.point(frame, com, (255,0,255), 7)

            if cop is not None:
                Draw.point(frame, cop, (0,255,0), 7)
                Draw.arrow(frame, cop, grf, c=(255,255,0))

            if pts["l_kn"] is not None:
                Draw.arrow(frame, pts["l_kn"], nf_l, c=(0,255,255))
                Draw.text(frame, f"T:{torque_l:.1f}", pts["l_kn"]+[0,20])

            if pts["r_kn"] is not None:
                Draw.arrow(frame, pts["r_kn"], nf_r, c=(0,255,255))
                Draw.text(frame, f"T:{torque_r:.1f}", pts["r_kn"]+[0,20])

            if pts["l_wr"] is not None:
                Draw.arrow(frame, pts["l_wr"], hf_l, c=(255,0,0))

            if pts["r_wr"] is not None:
                Draw.arrow(frame, pts["r_wr"], hf_r, c=(255,0,0))

            Draw.text(frame,
                      f"P{pid} {'STABLE' if stable else 'UNSTABLE'} {score:.2f}",
                      (30,40+pid*20),
                      (0,255,0) if stable else (0,0,255))

            if fall:
                Draw.text(frame, "FALL RISK", (200,40+pid*20), (0,0,255))

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

        frame = engine.process(frame)
        out.write(frame)

    cap.release()
    out.release()

    print("✅ Done:", OUTPUT_VIDEO)


if __name__ == "__main__":
    main()
