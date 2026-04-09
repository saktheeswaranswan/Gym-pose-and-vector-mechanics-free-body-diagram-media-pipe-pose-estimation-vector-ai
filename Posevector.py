# ---------------------------
# INSTALL
# ---------------------------
import os
os.system("pip install mediapipe opencv-python numpy")

import cv2
import numpy as np
import json
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request

# ---------------------------
# DOWNLOAD ONLY STABLE MODEL
# ---------------------------
MODEL_PATH = "pose.task"

if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
        MODEL_PATH
    )

# ---------------------------
# CONFIG
# ---------------------------
VIDEO_PATH = "video.mp4"
OUTPUT_VIDEO = "output.mp4"
OUTPUT_JSON = "poseesti.json"

VECTOR_SCALE = 120

# ---------------------------
# LOAD MODEL
# ---------------------------
pose = vision.PoseLandmarker.create_from_options(
    vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO
    )
)

# ---------------------------
# POSE EDGES (FULL BODY)
# ---------------------------
POSE_EDGES = [
    (0,1),(1,2),(2,3),(3,7),
    (0,4),(4,5),(5,6),(6,8),
    (9,10),
    (11,12),
    (11,13),(13,15),
    (12,14),(14,16),
    (11,23),(12,24),
    (23,24),
    (23,25),(25,27),(27,29),(29,31),
    (24,26),(26,28),(28,30),(30,32)
]

# ---------------------------
# HELPERS
# ---------------------------
def normalize(v):
    n = np.linalg.norm(v)
    return v/n if n!=0 else np.zeros_like(v)

def perpendicular(v):
    return np.array([-v[1], v[0]])

def get_point(lm,i,w,h):
    return np.array([lm[i].x*w, lm[i].y*h])

def draw_point(img,p):
    cv2.circle(img, tuple(p.astype(int)), 4, (0,255,0), -1)

def draw_edge(img,p1,p2):
    cv2.line(img, tuple(p1.astype(int)), tuple(p2.astype(int)), (255,0,0), 2)

def draw_vector(img,start,vec):
    end = (start + vec*VECTOR_SCALE).astype(int)
    cv2.arrowedLine(img, tuple(start.astype(int)), tuple(end), (0,0,255), 3)

def angle(a,b,c):
    ba=a-b
    bc=c-b
    cos=np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
    return np.degrees(np.arccos(np.clip(cos,-1,1)))

def draw_arc(img,center,ang):
    center = tuple(center.astype(int))
    cv2.ellipse(img, center, (40,40), 0, 0, int(ang), (0,255,255), 2)
    cv2.putText(img,f"{int(ang)}°",(center[0]+10,center[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),2)

# ---------------------------
# VIDEO
# ---------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
w = int(cap.get(3))
h = int(cap.get(4))

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,(w,h)
)

timeline=[]
frame_id=0

# ---------------------------
# LOOP
# ---------------------------
while cap.isOpened():
    ret,frame = cap.read()
    if not ret:
        break

    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,data=frame)
    t = int(frame_id/fps*1000)

    res = pose.detect_for_video(mp_img,t)

    data={"frame":frame_id,"pose":{},"angles":{},"vectors":{}}

    if res.pose_landmarks:

        lm = res.pose_landmarks[0]
        pts = [get_point(lm,i,w,h) for i in range(len(lm))]

        # DRAW ALL POINTS
        for p in pts:
            draw_point(frame,p)

        # DRAW FULL SKELETON
        for a,b in POSE_EDGES:
            draw_edge(frame,pts[a],pts[b])

        # ---------------------------
        # BIOMECHANICS
        # ---------------------------
        LHIP,LKNEE,LANKLE = pts[23],pts[25],pts[27]
        RHIP,RKNEE,RANKLE = pts[24],pts[26],pts[28]
        LHEEL,LFOOT = pts[29],pts[31]
        RHEEL,RFOOT = pts[30],pts[32]

        la = angle(LHIP,LKNEE,LANKLE)
        ra = angle(RHIP,RKNEE,RANKLE)

        draw_arc(frame,LKNEE,la)
        draw_arc(frame,RKNEE,ra)

        # FOOT VECTOR
        lf = normalize(LFOOT-LHEEL)
        rf = normalize(RFOOT-RHEEL)

        # PERPENDICULAR FORCE
        ln = normalize(perpendicular(lf))
        rn = normalize(perpendicular(rf))

        # force upward
        if ln[1]>0: ln*=-1
        if rn[1]>0: rn*=-1

        draw_vector(frame,LFOOT,ln)
        draw_vector(frame,RFOOT,rn)

        data["angles"]={"left_knee":float(la),"right_knee":float(ra)}
        data["vectors"]={
            "left_ground":ln.tolist(),
            "right_ground":rn.tolist()
        }

        data["pose"]={
            str(i):[lm[i].x,lm[i].y,lm[i].z]
            for i in range(len(lm))
        }

    timeline.append(data)
    out.write(frame)
    frame_id+=1

# ---------------------------
# SAVE
# ---------------------------
cap.release()
out.release()

with open(OUTPUT_JSON,"w") as f:
    json.dump(timeline,f,indent=2)

print("✅ DONE — FULL BODY + EDGES + VECTORS + JSON")
