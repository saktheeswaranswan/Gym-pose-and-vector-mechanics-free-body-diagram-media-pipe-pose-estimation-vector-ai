import cv2
import numpy as np
import math
from ultralytics import YOLO

# ---------------------------
# CONFIG
# ---------------------------
INPUT_VIDEO = "video.mp4"
OUTPUT_VIDEO = "inference_video.mp4"
MODEL_PATH = "yolo11n-pose.pt"

MIN_CONF = 0.25
SHOW_LIVE = False

MASS = 70.0
G = 9.81
MU = 0.6   # friction coefficient

# ---------------------------
# KEYPOINTS
# ---------------------------
KP = {
    "nose":0,"l_sh":5,"r_sh":6,"l_el":7,"r_el":8,"l_wr":9,"r_wr":10,
    "l_hp":11,"r_hp":12,"l_kn":13,"r_kn":14,"l_an":15,"r_an":16
}

# ---------------------------
# VECTOR
# ---------------------------
class Vec:
    @staticmethod
    def unit(v):
        n = np.linalg.norm(v)
        return v/n if n>1e-6 else np.zeros_like(v)

    @staticmethod
    def perp(v):
        return np.array([-v[1], v[0]], dtype=np.float32)

# ---------------------------
# BIOMECH
# ---------------------------
class Biomech:

    @staticmethod
    def foot(knee, ankle):
        v = ankle - knee
        axis = Vec.unit(v)
        normal = Vec.perp(axis)
        center = ankle
        return center, axis, normal

    @staticmethod
    def COM(kpts, conf):
        valid = [np.array(kpts[i]) for i in range(len(kpts)) if conf[i]>MIN_CONF]
        if not valid: return None
        return np.mean(valid, axis=0)

    @staticmethod
    def COP(com, support_pts):
        if not support_pts: return None
        xs = [p[0] for p in support_pts]
        y = max(p[1] for p in support_pts)
        return np.array([np.clip(com[0], min(xs), max(xs)), y], dtype=np.float32)

    @staticmethod
    def ground_forces(com, cop, feet):
        W = MASS * G

        if cop is None or com is None:
            return [], False, 0.0

        dx = com[0] - cop[0]
        h = abs(com[1] - cop[1]) + 1e-6

        lean_angle = math.atan2(abs(dx), h)

        Ft_total = W * math.tan(lean_angle)
        slip_ratio = Ft_total / (MU * W + 1e-6)

        forces = []

        for f in feet:
            center, axis, normal = f

            Fn = W / len(feet)
            Ft = min(Ft_total / len(feet), MU * Fn)

            traction_dir = -np.sign(dx) * Vec.unit(axis)
            normal_dir = np.array([0,-1], dtype=np.float32)

            forces.append({
                "center": center,
                "Fn": Fn,
                "Ft": Ft,
                "n_vec": normal_dir * Fn * 0.08,
                "t_vec": traction_dir * Ft * 0.08
            })

        stable = slip_ratio <= 1.0

        score = max(0.0, 1.0 - slip_ratio)

        return forces, stable, score

# ---------------------------
# DRAW
# ---------------------------
class Draw:
    def p(self,f,p,c=(0,255,255)):
        cv2.circle(f,(int(p[0]),int(p[1])),5,c,-1)

    def l(self,f,a,b,c=(200,200,200)):
        cv2.line(f,(int(a[0]),int(a[1])),(int(b[0]),int(b[1])),c,2)

    def arrow(self,f,p,v,c):
        e = p+v
        cv2.arrowedLine(f,(int(p[0]),int(p[1])),(int(e[0]),int(e[1])),c,2)

# ---------------------------
# ENGINE
# ---------------------------
class Engine:

    def __init__(self):
        self.model = YOLO(MODEL_PATH)
        self.d = Draw()

    def process(self, frame):

        res = self.model(frame, verbose=False)

        if not res or res[0].keypoints is None:
            return frame

        kpts_all = res[0].keypoints.xy.cpu().numpy()
        conf_all = res[0].keypoints.conf.cpu().numpy()

        for kpts, conf in zip(kpts_all, conf_all):

            pts = {k: (np.array(kpts[v]) if conf[v]>MIN_CONF else None) for k,v in KP.items()}

            # COM
            com = Biomech.COM(kpts, conf)

            # Feet
            feet = []
            support_pts = []

            if pts["l_kn"] is not None and pts["l_an"] is not None:
                f = Biomech.foot(pts["l_kn"], pts["l_an"])
                feet.append(f)
                support_pts.append(f[0])

            if pts["r_kn"] is not None and pts["r_an"] is not None:
                f = Biomech.foot(pts["r_kn"], pts["r_an"])
                feet.append(f)
                support_pts.append(f[0])

            # COP
            cop = Biomech.COP(com, support_pts)

            # Forces
            forces, stable, score = Biomech.ground_forces(com, cop, feet)

            # DRAW
            if com is not None:
                self.d.p(frame, com, (255,0,255))

            if cop is not None:
                self.d.p(frame, cop, (0,255,255))

            for f in forces:
                c = f["center"]

                # normal
                self.d.arrow(frame, c, f["n_vec"], (255,255,0))

                # traction
                self.d.arrow(frame, c, f["t_vec"], (0,165,255))

                # friction cone
                n = np.array([0,-1], dtype=np.float32)
                angle = math.atan(MU)

                left = Vec.unit([math.sin(angle), -math.cos(angle)])*50
                right = Vec.unit([-math.sin(angle), -math.cos(angle)])*50

                self.d.l(frame, c, c+left, (180,180,0))
                self.d.l(frame, c, c+right, (180,180,0))

            txt = f"STABLE" if stable else "SLIP"
            cv2.putText(frame, txt, (20,40), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0) if stable else (0,0,255),2)

        return frame

# ---------------------------
# MAIN
# ---------------------------
def main():

    cap = cv2.VideoCapture(INPUT_VIDEO)
    w = int(cap.get(3))
    h = int(cap.get(4))
    fps = cap.get(5) or 30

    out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))

    eng = Engine()

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = eng.process(frame)
        out.write(frame)

    cap.release()
    out.release()

if __name__ == "__main__":
    main()
