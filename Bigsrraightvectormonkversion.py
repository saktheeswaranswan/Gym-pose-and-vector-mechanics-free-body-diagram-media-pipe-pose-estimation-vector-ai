# ---------------------------
# INSTALL (Colab / Jupyter)
# ---------------------------
# !pip install mediapipe opencv-python numpy

import os
import json
import math
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------------------
# MODEL DOWNLOAD
# ---------------------------
MODEL_PATH = "pose_landmarker_full.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/1/"
    "pose_landmarker_full.task"
)

if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# ---------------------------
# CONFIG
# ---------------------------
VIDEO_PATH = "video.mp4"
OUTPUT_VIDEO = "output.mp4"
OUTPUT_JSON = "poseesti.json"

# Base scale for vectors
BASE_VECTOR_SCALE = 0.06

# ---------------------------
# LANDMARK INDEX MAP (MediaPipe Pose 33)
# ---------------------------
NOSE = 0
LEFT_EYE_INNER = 1
LEFT_EYE = 2
LEFT_EYE_OUTER = 3
RIGHT_EYE_INNER = 4
RIGHT_EYE = 5
RIGHT_EYE_OUTER = 6
LEFT_EAR = 7
RIGHT_EAR = 8

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

# ---------------------------
# SKELETON EDGES
# ---------------------------
POSE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
]

# ---------------------------
# HELPERS
# ---------------------------
def safe_norm(v):
    n = np.linalg.norm(v)
    if n < 1e-6:
        return np.zeros_like(v, dtype=np.float32)
    return v / n

def perpendicular(v):
    return np.array([-v[1], v[0]], dtype=np.float32)

def get_point(lm, i, w, h):
    return np.array([lm[i].x * w, lm[i].y * h], dtype=np.float32)

def get_visibility(lm, i):
    try:
        return float(lm[i].visibility)
    except Exception:
        return 1.0

def angle_deg(a, b, c):
    """
    Angle ABC in degrees, range 0..180.
    """
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
    cosv = np.dot(ba, bc) / denom
    return float(np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0))))

def draw_point(img, p, color=(0, 255, 0), r=4):
    cv2.circle(img, tuple(p.astype(int)), r, color, -1)

def draw_edge(img, p1, p2, color=(255, 0, 0), thickness=2):
    cv2.line(img, tuple(p1.astype(int)), tuple(p2.astype(int)), color, thickness)

def draw_text(img, text, p, color=(255, 255, 255)):
    x, y = int(p[0]), int(p[1])
    cv2.putText(
        img, text, (x + 8, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA
    )

def draw_angle_arc(img, a, b, c, radius=36, color=(0, 255, 255)):
    """
    Draw a visual arc around joint b.
    """
    center = tuple(b.astype(int))

    ang1 = (math.degrees(math.atan2(a[1] - b[1], a[0] - b[0])) + 360.0) % 360.0
    ang2 = (math.degrees(math.atan2(c[1] - b[1], c[0] - b[0])) + 360.0) % 360.0

    # Choose the shorter sweep for display
    diff = (ang2 - ang1) % 360.0
    if diff <= 180.0:
        start, end = ang1, ang2
    else:
        start, end = ang2, ang1

    cv2.ellipse(img, center, (radius, radius), 0, start, end, color, 2)

def body_center_from_points(pts, landmark_indices):
    valid = [pts[i] for i in landmark_indices if pts[i] is not None]
    if not valid:
        return np.array([0.0, 0.0], dtype=np.float32)
    return np.mean(np.array(valid, dtype=np.float32), axis=0)

def draw_biomech_joint_vector(
    img,
    joint_pt,
    proximal_pt,
    distal_pt,
    body_center,
    angle_value=None,
    base_scale=1.0,
    color=(0, 0, 255),
    thickness=3,
    label=None
):
    """
    Straight biomechanics-style reaction vector:
    - aligned with the limb axis (proximal -> distal)
    - longer as the joint approaches 180°
    - centered on the joint for a clean straight look
    """
    axis = safe_norm(distal_pt - proximal_pt)

    if np.linalg.norm(axis) < 1e-6:
        axis = safe_norm(distal_pt - joint_pt)

    if np.linalg.norm(axis) < 1e-6:
        axis = np.array([1.0, 0.0], dtype=np.float32)

    # Keep the vector pointing away from torso/body center for consistent visuals
    if np.dot(axis, joint_pt - body_center) < 0:
        axis *= -1.0

    # 0° = very bent, 180° = fully straight
    straightness = 1.0 if angle_value is None else np.clip(float(angle_value) / 180.0, 0.0, 1.0)

    # Make the vector clearly longer near full extension
    length_px = int(base_scale * (0.9 + 3.0 * straightness))
    length_px = max(18, length_px)

    # Draw a line through the joint so it looks straight and centered
    start = (joint_pt - axis * 0.55 * length_px).astype(int)
    end = (joint_pt + axis * 0.85 * length_px).astype(int)

    cv2.line(img, tuple(start), tuple(end), color, thickness)
    cv2.arrowedLine(img, tuple(joint_pt.astype(int)), tuple(end), color, thickness, tipLength=0.18)

    if label is not None:
        cv2.putText(
            img, label, (int(joint_pt[0]) + 6, int(joint_pt[1]) + 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA
        )

    return axis, length_px

def draw_terminal_biomech_vector(
    img,
    joint_pt,
    parent_pt,
    body_center,
    length_px=40,
    color=(0, 128, 255),
    thickness=2,
    label=None
):
    """
    For wrist / foot terminals:
    draw along the last bone direction, not perpendicular.
    """
    axis = safe_norm(joint_pt - parent_pt)

    if np.linalg.norm(axis) < 1e-6:
        axis = np.array([1.0, 0.0], dtype=np.float32)

    if np.dot(axis, joint_pt - body_center) < 0:
        axis *= -1.0

    start = joint_pt.astype(int)
    end = (joint_pt + axis * length_px).astype(int)

    cv2.arrowedLine(img, tuple(start), tuple(end), color, thickness, tipLength=0.25)

    if label is not None:
        cv2.putText(
            img, label, (int(joint_pt[0]) + 6, int(joint_pt[1]) + 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA
        )

    return axis, length_px

# ---------------------------
# MODEL LOAD
# ---------------------------
options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

pose = vision.PoseLandmarker.create_from_options(options)

# ---------------------------
# VIDEO IO
# ---------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

# ---------------------------
# JOINT SETS
# ---------------------------
ANGLE_JOINTS = {
    "left_shoulder":  (LEFT_ELBOW, LEFT_SHOULDER, LEFT_HIP),
    "right_shoulder": (RIGHT_ELBOW, RIGHT_SHOULDER, RIGHT_HIP),

    "left_elbow":     (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
    "right_elbow":    (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),

    "left_hip":       (LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE),
    "right_hip":      (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE),

    "left_knee":      (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
    "right_knee":     (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),

    "left_ankle":     (LEFT_KNEE, LEFT_ANKLE, LEFT_FOOT_INDEX),
    "right_ankle":    (RIGHT_KNEE, RIGHT_ANKLE, RIGHT_FOOT_INDEX),
}

TERMINAL_VECTORS = {
    "left_wrist":  (LEFT_ELBOW, LEFT_WRIST),
    "right_wrist": (RIGHT_ELBOW, RIGHT_WRIST),
    "left_foot":   (LEFT_ANKLE, LEFT_FOOT_INDEX),
    "right_foot":  (RIGHT_ANKLE, RIGHT_FOOT_INDEX),
}

# ---------------------------
# MAIN LOOP
# ---------------------------
timeline = []
frame_id = 0

while cap.isOpened():
    ret, frame_bgr = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    timestamp_ms = int((frame_id / fps) * 1000)
    result = pose.detect_for_video(mp_image, timestamp_ms)

    data = {
        "frame": frame_id,
        "angles": {},
        "vectors": {},
        "pose": {}
    }

    if result.pose_landmarks:
        lm = result.pose_landmarks[0]
        pts = [get_point(lm, i, w, h) for i in range(len(lm))]

        vis_ok = [i for i in range(len(lm)) if get_visibility(lm, i) > 0.2]

        body_center = body_center_from_points(pts, [
            LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP
        ])

        # Draw landmarks
        for i in vis_ok:
            draw_point(frame_bgr, pts[i], color=(0, 255, 0), r=4)

        # Draw skeleton
        for a, b in POSE_EDGES:
            if a < len(pts) and b < len(pts):
                draw_edge(frame_bgr, pts[a], pts[b], color=(255, 0, 0), thickness=2)

        # Angle joints + straight biomech vectors
        base_px = int(min(w, h) * BASE_VECTOR_SCALE)

        for name, (a_idx, b_idx, c_idx) in ANGLE_JOINTS.items():
            if a_idx >= len(pts) or b_idx >= len(pts) or c_idx >= len(pts):
                continue

            if (
                get_visibility(lm, a_idx) < 0.2
                or get_visibility(lm, b_idx) < 0.2
                or get_visibility(lm, c_idx) < 0.2
            ):
                continue

            a, b, c = pts[a_idx], pts[b_idx], pts[c_idx]
            ang = angle_deg(a, b, c)

            # Angle arc at joint
            draw_angle_arc(frame_bgr, a, b, c, radius=34, color=(0, 255, 255))

            # Straight reaction vector aligned with limb
            axis, vec_len = draw_biomech_joint_vector(
                frame_bgr,
                joint_pt=b,
                proximal_pt=a,
                distal_pt=c,
                body_center=body_center,
                angle_value=ang,
                base_scale=base_px,
                color=(0, 0, 255),
                thickness=3,
                label=f"{name}: {int(ang)}°"
            )

            data["angles"][name] = float(ang)
            data["vectors"][name] = {
                "axis": axis.tolist(),
                "length_px": int(vec_len)
            }

        # Terminal vectors for wrists and feet
        for name, (parent_idx, joint_idx) in TERMINAL_VECTORS.items():
            if parent_idx >= len(pts) or joint_idx >= len(pts):
                continue

            if get_visibility(lm, parent_idx) < 0.2 or get_visibility(lm, joint_idx) < 0.2:
                continue

            parent_pt = pts[parent_idx]
            joint_pt = pts[joint_idx]

            axis, vec_len = draw_terminal_biomech_vector(
                frame_bgr,
                joint_pt=joint_pt,
                parent_pt=parent_pt,
                body_center=body_center,
                length_px=max(24, int(base_px * 1.25)),
                color=(0, 128, 255),
                thickness=2,
                label=name
            )

            data["vectors"][name] = {
                "axis": axis.tolist(),
                "length_px": int(vec_len)
            }

        # Save raw pose landmarks
        data["pose"] = {
            str(i): [float(lm[i].x), float(lm[i].y), float(lm[i].z)]
            for i in range(len(lm))
        }

    timeline.append(data)
    out.write(frame_bgr)
    frame_id += 1

# ---------------------------
# SAVE OUTPUT
# ---------------------------
cap.release()
out.release()
pose.close()

with open(OUTPUT_JSON, "w") as f:
    json.dump(timeline, f, indent=2)

print("✅ DONE — full skeleton + points + angle arcs + straight biomechanical vectors + JSON saved")
