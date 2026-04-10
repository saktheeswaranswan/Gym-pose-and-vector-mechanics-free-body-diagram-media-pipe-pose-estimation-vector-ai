# ---------------------------
# INSTALL
# ---------------------------
# !pip install mediapipe opencv-python numpy

import os, json, math, urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------------------
# CONFIG
# ---------------------------
VIDEO_PATH = "video.mp4"
OUTPUT_VIDEO = "output.mp4"
OUTPUT_JSON = "pose_physics.json"

MASS = 20.0
GRAVITY = 9.81
BODY_FORCE = MASS * GRAVITY   # 196.2 N

BASE_VECTOR_SCALE = 0.08

# ---------------------------
# MODEL
# ---------------------------
MODEL_PATH = "pose_landmarker_full.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"

if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# ---------------------------
# LANDMARKS
# ---------------------------
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_FOOT, RIGHT_FOOT = 31, 32

# ---------------------------
# HELPERS
# ---------------------------
def norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else np.zeros_like(v)

def perp(v):
    return np.array([-v[1], v[0]], dtype=np.float32)

def angle(a, b, c):
    ba = a - b
    bc = c - b
    return np.degrees(np.arccos(np.clip(np.dot(norm(ba), norm(bc)), -1, 1)))

def pt(lm, i, w, h):
    return np.array([lm[i].x * w, lm[i].y * h], dtype=np.float32)

# ---------------------------
# PHYSICS CORE
# ---------------------------
def reaction_force(a, b, c):
    limb = norm(c - a)
    n = norm(perp(limb))
    theta = np.radians(angle(a, b, c))

    magnitude = BODY_FORCE * abs(np.sin(theta))
    return n * magnitude, magnitude

def torque(joint, force_vec, ref):
    r = joint - ref
    return np.cross(r, force_vec)

def center_of_mass(pts):
    return np.mean(pts, axis=0)

def center_of_pressure(lf, rf):
    return (lf + rf) / 2

def balance_check(com, lf, rf):
    min_x = min(lf[0], rf[0])
    max_x = max(lf[0], rf[0])
    return min_x <= com[0] <= max_x

# ---------------------------
# MODEL LOAD
# ---------------------------
options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO
)

pose = vision.PoseLandmarker.create_from_options(options)

# ---------------------------
# VIDEO
# ---------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
w, h = int(cap.get(3)), int(cap.get(4))

out = cv2.VideoWriter(OUTPUT_VIDEO,
                      cv2.VideoWriter_fourcc(*"mp4v"),
                      fps, (w, h))

timeline = []
frame_id = 0

# ---------------------------
# MAIN LOOP
# ---------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    res = pose.detect_for_video(mp_img, int(frame_id * 1000 / fps))

    data = {"frame": frame_id}

    if res.pose_landmarks:
        lm = res.pose_landmarks[0]
        pts = [pt(lm, i, w, h) for i in range(len(lm))]

        # ---- KEY POINTS ----
        lhip, rhip = pts[LEFT_HIP], pts[RIGHT_HIP]
        lk, rk = pts[LEFT_KNEE], pts[RIGHT_KNEE]
        la, ra = pts[LEFT_ANKLE], pts[RIGHT_ANKLE]
        lf, rf = pts[LEFT_FOOT], pts[RIGHT_FOOT]

        # ---- COM ----
        com = center_of_mass([lhip, rhip, lk, rk])

        # ---- CoP ----
        cop = center_of_pressure(lf, rf)

        # ---- BALANCE ----
        stable = balance_check(com, lf, rf)

        # ---- FORCES ----
        knee_force_L, magL = reaction_force(lhip, lk, la)
        knee_force_R, magR = reaction_force(rhip, rk, ra)

        # ---- TORQUE ----
        torque_L = torque(lk, knee_force_L, lhip)
        torque_R = torque(rk, knee_force_R, rhip)

        # ---- DRAW ----
        scale = BASE_VECTOR_SCALE * min(w, h)

        def draw_vec(p, v, color):
            end = (p + norm(v) * scale).astype(int)
            cv2.arrowedLine(frame, tuple(p.astype(int)), tuple(end), color, 3)

        draw_vec(lk, knee_force_L, (0,0,255))
        draw_vec(rk, knee_force_R, (0,0,255))

        # COM & CoP
        cv2.circle(frame, tuple(com.astype(int)), 6, (255,255,0), -1)
        cv2.circle(frame, tuple(cop.astype(int)), 6, (0,255,255), -1)

        # BALANCE TEXT
        status = "STABLE" if stable else "FALL RISK"
        cv2.putText(frame, status, (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0,255,0) if stable else (0,0,255), 3)

        data.update({
            "COM": com.tolist(),
            "CoP": cop.tolist(),
            "stable": stable,
            "force_L": float(magL),
            "force_R": float(magR),
            "torque_L": float(torque_L),
            "torque_R": float(torque_R)
        })

    timeline.append(data)
    out.write(frame)
    frame_id += 1

# ---------------------------
# SAVE
# ---------------------------
cap.release()
out.release()
pose.close()

with open(OUTPUT_JSON, "w") as f:
    json.dump(timeline, f, indent=2)

print("🔥 DONE — FULL PHYSICS ENGINE ACTIVE")
