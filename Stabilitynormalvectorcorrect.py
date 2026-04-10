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
BODY_FORCE = MASS * GRAVITY  # 196.2 N

BASE_VECTOR_SCALE = 0.10

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
# TRUE BIOMECHANICS
# ---------------------------
def joint_normal_force(a, b, c, body_center):
    limb = norm(c - a)
    normal = norm(perp(limb))

    # Ensure outward consistency
    if np.dot(normal, b - body_center) < 0:
        normal *= -1

    theta = np.radians(angle(a, b, c))
    magnitude = BODY_FORCE * abs(np.sin(theta))

    return normal * magnitude, magnitude

def ground_reaction_force(foot, ankle):
    foot_dir = norm(foot - ankle)
    normal = norm(perp(foot_dir))

    # force upward
    if normal[1] > 0:
        normal *= -1

    magnitude = BODY_FORCE * 0.5
    return normal * magnitude, magnitude

def torque(joint, force_vec, ref):
    r = joint - ref
    return float(np.cross(r, force_vec))

def center_of_mass(points):
    return np.mean(points, axis=0)

def center_of_pressure(lf, rf):
    return (lf + rf) / 2

def is_balanced(com, lf, rf):
    return min(lf[0], rf[0]) <= com[0] <= max(lf[0], rf[0])

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

        # Points
        lhip, rhip = pts[LEFT_HIP], pts[RIGHT_HIP]
        lk, rk = pts[LEFT_KNEE], pts[RIGHT_KNEE]
        la, ra = pts[LEFT_ANKLE], pts[RIGHT_ANKLE]
        lf, rf = pts[LEFT_FOOT], pts[RIGHT_FOOT]

        body_center = center_of_mass([lhip, rhip])

        # Forces
        fL, magL = joint_normal_force(lhip, lk, la, body_center)
        fR, magR = joint_normal_force(rhip, rk, ra, body_center)

        grfL, _ = ground_reaction_force(lf, la)
        grfR, _ = ground_reaction_force(rf, ra)

        # Torque
        tauL = torque(lk, fL, lhip)
        tauR = torque(rk, fR, rhip)

        # COM & CoP
        com = center_of_mass([lhip, rhip, lk, rk])
        cop = center_of_pressure(lf, rf)

        stable = is_balanced(com, lf, rf)

        # DRAW
        scale = BASE_VECTOR_SCALE * min(w, h)

        def draw_vec(p, v, color):
            end = (p + norm(v) * scale).astype(int)
            cv2.arrowedLine(frame, tuple(p.astype(int)), tuple(end), color, 3)

        # Joint forces
        draw_vec(lk, fL, (0,0,255))
        draw_vec(rk, fR, (0,0,255))

        # Ground forces
        draw_vec(lf, grfL, (0,255,255))
        draw_vec(rf, grfR, (0,255,255))

        # COM & CoP
        cv2.circle(frame, tuple(com.astype(int)), 6, (255,255,0), -1)
        cv2.circle(frame, tuple(cop.astype(int)), 6, (0,255,255), -1)

        # Balance
        text = "STABLE" if stable else "FALL RISK"
        cv2.putText(frame, text, (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0,255,0) if stable else (0,0,255), 3)

        data.update({
            "force_L": float(magL),
            "force_R": float(magR),
            "torque_L": tauL,
            "torque_R": tauR,
            "COM": com.tolist(),
            "CoP": cop.tolist(),
            "stable": stable
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

print("🔥 TRUE BIOMECHANICS SYSTEM COMPLETE")
