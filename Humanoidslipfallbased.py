# ---------------------------
# INSTALL (RUN ONCE)
# ---------------------------
# !pip install ultralytics opencv-python numpy -q

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
OUTPUT_VIDEO = "output.mp4"
OUTPUT_CSV = "timeline.csv"
MODEL_PATH = "yolo11n-pose.pt"

MIN_CONF = 0.25
HISTORY = 10

MASS_KG = 70
GRAVITY = 9.81
MU = 0.65

# ---------------------------
# KEYPOINTS
# ---------------------------
KP = {
    "l_sh": 5, "r_sh": 6,
    "l_hp": 11, "r_hp": 12,
    "l_kn": 13, "r_kn": 14,
    "l_an": 15, "r_an": 16
}

# ---------------------------
# VECTOR UTILS
# ---------------------------
class Vec:
    @staticmethod
    def norm(v):
        return np.linalg.norm(v)

    @staticmethod
    def unit(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else np.zeros(2)

    @staticmethod
    def perp(v):
        return np.array([-v[1], v[0]])

# ---------------------------
# BIOMECH
# ---------------------------
class Biomech:

    @staticmethod
    def estimate_com(kpts, conf):
        pts = []
        for k in ["l_sh","r_sh","l_hp","r_hp"]:
            if conf[KP[k]] > MIN_CONF:
                pts.append(kpts[KP[k]])
        if len(pts) < 2:
            return None
        return np.mean(pts, axis=0)

    @staticmethod
    def support_polygon(kpts, conf):
        pts = []
        for k in ["l_an","r_an"]:
            if conf[KP[k]] > MIN_CONF:
                pts.append(kpts[KP[k]])
        if len(pts) < 1:
            return None
        pts = np.array(pts)
        if len(pts) == 1:
            return pts
        return cv2.convexHull(pts.astype(np.float32)).reshape(-1,2)

    @staticmethod
    def projection_stability(com, hull):
        if com is None or hull is None:
            return False, None, 0, 0

        ground_y = np.max(hull[:,1])
        proj = np.array([com[0], ground_y])

        if len(hull) >= 3:
            inside = cv2.pointPolygonTest(hull.reshape(-1,1,2), tuple(proj), False) >= 0
        else:
            xmin, xmax = np.min(hull[:,0]), np.max(hull[:,0])
            inside = xmin <= proj[0] <= xmax

        # margin
        def dist(a,b,p):
            ap = p-a
            ab = b-a
            t = np.clip(np.dot(ap,ab)/(np.dot(ab,ab)+1e-6),0,1)
            return np.linalg.norm(p-(a+t*ab))

        if len(hull)>=3:
            margin = min(dist(hull[i],hull[(i+1)%len(hull)],proj) for i in range(len(hull)))
        else:
            margin = abs(proj[0]-np.mean(hull[:,0]))

        width = max(30,np.max(hull[:,0])-np.min(hull[:,0]))
        score = min(1, margin/(0.2*width))

        return inside, proj, margin, score

# ---------------------------
# ENGINE
# ---------------------------
class Engine:

    def __init__(self):
        self.model = YOLO(MODEL_PATH)
        self.history = {}
        self.tracks = {}
        self.next_id = 0

    # ---------------------------
    # TRACKING
    # ---------------------------
    def assign_id(self, com):
        if com is None:
            return None

        best_id = None
        best_dist = 9999

        for pid, prev in self.tracks.items():
            d = np.linalg.norm(prev - com)
            if d < best_dist and d < 80:
                best_dist = d
                best_id = pid

        if best_id is None:
            best_id = self.next_id
            self.next_id += 1

        self.tracks[best_id] = com.copy()
        return best_id

    def get_hist(self, pid):
        if pid not in self.history:
            self.history[pid] = deque(maxlen=HISTORY)
        return self.history[pid]

    # ---------------------------
    # FALL PREDICTION
    # ---------------------------
    def predict_fall(self, pid, com, stable, slip):
        hist = self.get_hist(pid)
        hist.append(com.copy())

        if len(hist) < 3:
            return False,0

        pts = np.array(hist)
        v = pts[1:] - pts[:-1]

        speed = np.mean(np.linalg.norm(v,axis=1))
        trend = np.mean(v[:,0])

        risk = 0
        risk += min(1,speed/12)*0.4
        risk += min(1,abs(trend)/6)*0.3
        risk += 0.2 if not stable else 0
        risk += min(1,max(0,slip-1))*0.3

        return risk>0.55, risk

    # ---------------------------
    # PROCESS
    # ---------------------------
    def process_person(self, frame, kpts, conf, frame_idx):

        com = Biomech.estimate_com(kpts, conf)
        pid = self.assign_id(com)

        if pid is None:
            return frame,None

        hull = Biomech.support_polygon(kpts, conf)

        # projection stability
        inside, proj, margin, mscore = Biomech.projection_stability(com, hull)

        # simple friction
        slip = abs(com[0]-proj[0]) / 50 if proj is not None else 0

        final = 0.4*(1-slip) + 0.3*mscore + 0.3*(1 if inside else 0)
        stable = final > 0.55

        fall, fr = self.predict_fall(pid, com, stable, slip)

        # DRAW
        if com is not None:
            cv2.circle(frame, tuple(com.astype(int)),6,(255,0,255),-1)

        if proj is not None:
            cv2.line(frame, tuple(com.astype(int)), tuple(proj.astype(int)), (255,0,255),2)
            cv2.circle(frame, tuple(proj.astype(int)),5,(0,255,255),-1)

        if hull is not None and len(hull)>=3:
            overlay = frame.copy()
            cv2.fillPoly(overlay,[hull.astype(int)],(0,255,0))
            frame = cv2.addWeighted(overlay,0.2,frame,0.8,0)

        color = (0,255,0) if stable else (0,0,255)

        cv2.putText(frame,f"ID {pid}",(20,40+pid*20),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

        cv2.putText(frame,f"S:{final:.2f} M:{margin:.1f}",
                    (20,60+pid*20),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1)

        if fall:
            cv2.putText(frame,"FALL",(400,50+pid*30),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

        row = {
            "frame":frame_idx,
            "id":pid,
            "stable":stable,
            "score":final,
            "margin":margin,
            "fall":fall
        }

        return frame,row

    def process_frame(self, frame, idx):
        results = self.model(frame, verbose=False)

        if not results or results[0].keypoints is None:
            return frame,[]

        kpts_all = results[0].keypoints.xy.cpu().numpy()
        conf_all = results[0].keypoints.conf.cpu().numpy()

        rows = []

        for kpts, conf in zip(kpts_all, conf_all):
            frame,row = self.process_person(frame,kpts,conf,idx)
            if row:
                rows.append(row)

        return frame,rows

# ---------------------------
# MAIN
# ---------------------------
def main():

    cap = cv2.VideoCapture(INPUT_VIDEO)
    w = int(cap.get(3))
    h = int(cap.get(4))
    fps = cap.get(5) or 30

    out = cv2.VideoWriter(OUTPUT_VIDEO,
                          cv2.VideoWriter_fourcc(*"mp4v"),
                          fps,(w,h))

    engine = Engine()
    all_rows=[]
    idx=0

    while True:
        ret,frame = cap.read()
        if not ret:
            break

        frame,rows = engine.process_frame(frame,idx)
        all_rows.extend(rows)

        out.write(frame)
        idx+=1

    cap.release()
    out.release()

    # CSV
    if all_rows:
        with open(OUTPUT_CSV,"w",newline="") as f:
            writer = csv.DictWriter(f,fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)

    print("DONE")

if __name__ == "__main__":
    main()
