# ---------------------------
# INSTALL
# ---------------------------
# !pip install ultralytics opencv-python numpy -q

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

SHOW_LIVE = True          # Set False in Colab if imshow is not supported
DRAW_SUPPORTHULL = True
DRAW_NORMALS = True
DRAW_COM_COG = True
DRAW_REACTION_VECTORS = True

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

# Joint triplets: (proximal, joint, distal)
JOINTS = {
    "left_elbow":  (KP["l_sh"], KP["l_el"], KP["l_wr"]),
    "right_elbow": (KP["r_sh"], KP["r_el"], KP["r_wr"]),
    "left_knee":   (KP["l_hp"], KP["l_kn"], KP["l_an"]),
    "right_knee":  (KP["r_hp"], KP["r_kn"], KP["r_an"]),
    "left_hip":    (KP["l_sh"], KP["l_hp"], KP["l_kn"]),
    "right_hip":   (KP["r_sh"], KP["r_hp"], KP["r_kn"]),
    # optional shoulder angles
    "left_shoulder":  (KP["l_el"], KP["l_sh"], KP["l_hp"]),
    "right_shoulder": (KP["r_el"], KP["r_sh"], KP["r_hp"]),
}

# Approximate segment mass weights (normalized later)
SEGMENT_WEIGHTS = {
    "head": 0.08,
    "trunk": 0.50,
    "upper_arm_l": 0.03,
    "upper_arm_r": 0.03,
    "forearm_l": 0.02,
    "forearm_r": 0.02,
    "thigh_l": 0.10,
    "thigh_r": 0.10,
    "shank_l": 0.0465,
    "shank_r": 0.0465,
    "foot_l": 0.0145,
    "foot_r": 0.0145,
}

# ---------------------------
# MATH HELPERS
# ---------------------------
class Vec:
    @staticmethod
    def norm(v):
        return float(np.linalg.norm(v))

    @staticmethod
    def unit(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-8 else np.zeros_like(v, dtype=np.float32)

    @staticmethod
    def perp(v):
        return np.array([-v[1], v[0]], dtype=np.float32)

    @staticmethod
    def mid(a, b):
        return (a + b) * 0.5

    @staticmethod
    def angle(a, b, c):
        ba = Vec.unit(a - b)
        bc = Vec.unit(c - b)
        val = np.clip(np.dot(ba, bc), -1.0, 1.0)
        return float(np.degrees(np.arccos(val)))

    @staticmethod
    def rotate(v, deg):
        r = np.radians(deg)
        c, s = np.cos(r), np.sin(r)
        return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]], dtype=np.float32)

def pt(x):
    return tuple(np.round(x).astype(int))

def valid_point(conf, idx, min_conf=MIN_CONF):
    return conf[idx] > min_conf

def get_point(kpts, conf, idx):
    return kpts[idx].astype(np.float32) if valid_point(conf, idx) else None

def safe_mean(points):
    pts = [p for p in points if p is not None]
    if not pts:
        return None
    return np.mean(np.array(pts, dtype=np.float32), axis=0)

def clip01(x):
    return float(max(0.0, min(1.0, x)))

# ---------------------------
# GEOMETRY HELPERS
# ---------------------------
def convex_hull(points):
    pts = np.array(points, dtype=np.float32)
    if len(pts) <= 1:
        return pts

    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))

    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))

    hull = lower[:-1] + upper[:-1]
    return np.array(hull, dtype=np.float32)

def point_in_poly(point, poly):
    if poly is None or len(poly) < 3:
        return False
    x, y = point
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        intersect = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-8) + xi
        )
        if intersect:
            inside = not inside
        j = i
    return inside

def point_to_segment_distance(p, a, b):
    ap = p - a
    ab = b - a
    denom = np.dot(ab, ab) + 1e-8
    t = np.clip(np.dot(ap, ab) / denom, 0.0, 1.0)
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))

def distance_to_polygon_edges(p, poly):
    if poly is None or len(poly) < 2:
        return 0.0
    dmin = 1e9
    for i in range(len(poly)):
        a = poly[i]
        b = poly[(i + 1) % len(poly)]
        dmin = min(dmin, point_to_segment_distance(p, a, b))
    return float(dmin)

# ---------------------------
# DRAWING
# ---------------------------
class Drawer:
    def point(self, frame, p, color=(0, 255, 255), r=4):
        cv2.circle(frame, pt(p), r, color, -1)

    def line(self, frame, p1, p2, color=(200, 200, 200), thickness=2):
        cv2.line(frame, pt(p1), pt(p2), color, thickness)

    def arrow(self, frame, p, v, color=(255, 0, 0), txt=None, scale=1.0):
        end = p + v * scale
        cv2.arrowedLine(
            frame,
            pt(p),
            pt(end),
            color,
            3,
            tipLength=0.25
        )
        if txt is not None:
            cv2.putText(
                frame,
                txt,
                pt(end + np.array([4, -4], dtype=np.float32)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA
            )

    def arc(self, frame, a, b, c, color=(0, 255, 255)):
        ang1 = math.atan2(a[1] - b[1], a[0] - b[0])
        ang2 = math.atan2(c[1] - b[1], c[0] - b[0])
        diff = (ang2 - ang1 + math.pi) % (2 * math.pi) - math.pi

        pts = []
        for t in np.linspace(0, 1, 20):
            ang = ang1 + diff * t
            x = int(b[0] + 28 * math.cos(ang))
            y = int(b[1] + 28 * math.sin(ang))
            pts.append([x, y])

        pts = np.array(pts, np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], False, color, 2)

    def label(self, frame, text, p, color=(255, 255, 255), scale=0.55, thickness=1):
        cv2.putText(frame, text, pt(p), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

# ---------------------------
# BIOMECHANICS
# ---------------------------
class Biomech:
    @staticmethod
    def estimate_foot_polygon(knee, ankle, side="left"):
        """
        Creates a pseudo-foot polygon from knee->ankle shank direction.
        This is a heuristic because COCO does not provide toe/heel keypoints.
        """
        shank = ankle - knee
        L = Vec.norm(shank)
        if L < 1e-6:
            return None, None, None

        shank_u = Vec.unit(shank)
        normal = Vec.perp(shank_u)

        # Side-dependent lateral bias to keep left/right foot distinct visually
        side_bias = -1.0 if side == "left" else 1.0
        lateral = normal * side_bias

        foot_len = max(18.0, 0.45 * L)
        foot_wid = max(8.0, 0.20 * foot_len)

        heel = ankle - 0.25 * foot_len * shank_u
        toe = ankle + 0.85 * foot_len * shank_u

        heel_l = heel + foot_wid * lateral
        heel_r = heel - foot_wid * lateral
        toe_l = toe + foot_wid * lateral
        toe_r = toe - foot_wid * lateral

        poly = np.array([heel_l, heel_r, toe_r, toe_l], dtype=np.float32)
        center = np.mean(poly, axis=0)

        # Foot axis and local normal
        foot_axis = Vec.unit(toe - heel)
        foot_normal = Vec.perp(foot_axis)

        return poly, center, (foot_axis, foot_normal)

    @staticmethod
    def joint_reaction_vector(a, b, c, com=None, base=60.0):
        """
        Heuristic biomechanics vector:
        - direction changes with joint angle
        - magnitude increases with flexion/extension deviation and COM bias
        """
        u1 = Vec.unit(a - b)   # proximal direction toward proximal joint
        u2 = Vec.unit(c - b)   # distal direction toward distal joint
        ang = Vec.angle(a, b, c)

        flex = abs(180.0 - ang) / 180.0  # 0 straight, 1 highly bent
        align = abs(float(np.dot(u1, u2)))  # 1 aligned, 0 orthogonal, -1 opposite

        bisector = u1 + u2
        if Vec.norm(bisector) < 1e-6:
            bisector = Vec.perp(u2)
        bisector = Vec.unit(bisector)

        distal_normal = Vec.perp(u2)  # normal to distal segment
        gravity = np.array([0.0, -1.0], dtype=np.float32)

        if com is not None and Vec.norm(com - b) > 1e-6:
            com_bias = Vec.unit(com - b)
        else:
            com_bias = np.zeros(2, dtype=np.float32)

        # Weighted direction: joint geometry + segment normal + gravity + COM bias
        direction = Vec.unit(
            0.38 * bisector +
            0.28 * distal_normal +
            0.18 * gravity +
            0.16 * com_bias
        )
        if Vec.norm(direction) < 1e-6:
            direction = Vec.unit(distal_normal if Vec.norm(distal_normal) > 1e-6 else gravity)

        # Magnitude increases with bend, misalignment, and COM offset from joint normal
        com_load = 0.0
        if com is not None and Vec.norm(com - b) > 1e-6:
            com_dir = Vec.unit(com - b)
            com_load = abs(float(np.dot(com_dir, distal_normal)))

        mag = base + 140.0 * flex + 35.0 * (1.0 - align) + 55.0 * com_load
        return direction * mag, float(mag), float(ang), direction, float(flex), float(com_load)

    @staticmethod
    def estimate_com(kpts, conf):
        """
        Approximate COM/COG using weighted segment midpoints.
        """
        shoulders = safe_mean([get_point(kpts, conf, KP["l_sh"]), get_point(kpts, conf, KP["r_sh"])])
        hips = safe_mean([get_point(kpts, conf, KP["l_hp"]), get_point(kpts, conf, KP["r_hp"])])
        nose = get_point(kpts, conf, KP["nose"])

        l_sh = get_point(kpts, conf, KP["l_sh"])
        r_sh = get_point(kpts, conf, KP["r_sh"])
        l_el = get_point(kpts, conf, KP["l_el"])
        r_el = get_point(kpts, conf, KP["r_el"])
        l_wr = get_point(kpts, conf, KP["l_wr"])
        r_wr = get_point(kpts, conf, KP["r_wr"])
        l_hp = get_point(kpts, conf, KP["l_hp"])
        r_hp = get_point(kpts, conf, KP["r_hp"])
        l_kn = get_point(kpts, conf, KP["l_kn"])
        r_kn = get_point(kpts, conf, KP["r_kn"])
        l_an = get_point(kpts, conf, KP["l_an"])
        r_an = get_point(kpts, conf, KP["r_an"])

        segments = []

        if nose is not None and shoulders is not None:
            head_mid = Vec.mid(nose, shoulders)
            segments.append((head_mid, SEGMENT_WEIGHTS["head"]))

        if shoulders is not None and hips is not None:
            trunk_mid = Vec.mid(shoulders, hips)
            segments.append((trunk_mid, SEGMENT_WEIGHTS["trunk"]))

        if l_sh is not None and l_el is not None:
            segments.append((Vec.mid(l_sh, l_el), SEGMENT_WEIGHTS["upper_arm_l"]))
        if r_sh is not None and r_el is not None:
            segments.append((Vec.mid(r_sh, r_el), SEGMENT_WEIGHTS["upper_arm_r"]))
        if l_el is not None and l_wr is not None:
            segments.append((Vec.mid(l_el, l_wr), SEGMENT_WEIGHTS["forearm_l"]))
        if r_el is not None and r_wr is not None:
            segments.append((Vec.mid(r_el, r_wr), SEGMENT_WEIGHTS["forearm_r"]))

        if l_hp is not None and l_kn is not None:
            segments.append((Vec.mid(l_hp, l_kn), SEGMENT_WEIGHTS["thigh_l"]))
        if r_hp is not None and r_kn is not None:
            segments.append((Vec.mid(r_hp, r_kn), SEGMENT_WEIGHTS["thigh_r"]))

        if l_kn is not None and l_an is not None:
            segments.append((Vec.mid(l_kn, l_an), SEGMENT_WEIGHTS["shank_l"]))
        if r_kn is not None and r_an is not None:
            segments.append((Vec.mid(r_kn, r_an), SEGMENT_WEIGHTS["shank_r"]))

        # Pseudo-foot segments
        if l_kn is not None and l_an is not None:
            poly, center, _ = Biomech.estimate_foot_polygon(l_kn, l_an, "left")
            if center is not None:
                segments.append((center, SEGMENT_WEIGHTS["foot_l"]))
        if r_kn is not None and r_an is not None:
            poly, center, _ = Biomech.estimate_foot_polygon(r_kn, r_an, "right")
            if center is not None:
                segments.append((center, SEGMENT_WEIGHTS["foot_r"]))

        if not segments:
            return None

        pts = np.array([p for p, _ in segments], dtype=np.float32)
        ws = np.array([w for _, w in segments], dtype=np.float32)
        ws = ws / (np.sum(ws) + 1e-8)
        com = np.sum(pts * ws[:, None], axis=0)
        return com

    @staticmethod
    def support_polygon(kpts, conf):
        """
        Build support points from both pseudo-feet and ankles.
        """
        support_pts = []

        # left
        l_kn = get_point(kpts, conf, KP["l_kn"])
        l_an = get_point(kpts, conf, KP["l_an"])
        if l_kn is not None and l_an is not None:
            poly, _, _ = Biomech.estimate_foot_polygon(l_kn, l_an, "left")
            if poly is not None:
                support_pts.extend(list(poly))
        elif l_an is not None:
            support_pts.append(l_an)

        # right
        r_kn = get_point(kpts, conf, KP["r_kn"])
        r_an = get_point(kpts, conf, KP["r_an"])
        if r_kn is not None and r_an is not None:
            poly, _, _ = Biomech.estimate_foot_polygon(r_kn, r_an, "right")
            if poly is not None:
                support_pts.extend(list(poly))
        elif r_an is not None:
            support_pts.append(r_an)

        if len(support_pts) < 2:
            return None, support_pts

        hull = convex_hull(support_pts)
        return hull, support_pts

    @staticmethod
    def stability_score(com, support_hull, support_pts, frame_w):
        """
        Heuristic stability score:
        - Inside support hull => more stable
        - Larger margin to edges => more stable
        - Wider stance => more stable
        """
        if com is None:
            return False, 0.0, None

        if support_hull is None or len(support_hull) < 2:
            # fallback to ankle width
            if len(support_pts) >= 2:
                xs = [p[0] for p in support_pts]
                xmin, xmax = min(xs), max(xs)
                width = max(20.0, xmax - xmin)
                center_x = 0.5 * (xmin + xmax)
                dist = abs(float(com[0] - center_x))
                score = math.exp(-dist / (0.35 * width + 1e-6))
                stable = score >= 0.55
                return stable, float(score), None
            return False, 0.0, None

        # Determine support center and spread
        center = np.mean(support_hull, axis=0)
        edge_dist = distance_to_polygon_edges(com, support_hull)
        inside = point_in_poly(com, support_hull)

        # x-offset from support center is the main balance indicator in 2D video
        xs = support_hull[:, 0]
        width = max(30.0, float(np.max(xs) - np.min(xs)))

        x_dist = abs(float(com[0] - center[0]))
        x_score = math.exp(-x_dist / (0.35 * width + 1e-6))

        # edge margin normalized by width
        margin_score = clip01(edge_dist / (0.18 * width + 1e-6))

        # Combine
        score = (0.58 * x_score) + (0.42 * margin_score)
        if inside:
            score = min(1.0, score + 0.15)
        else:
            score *= 0.75

        stable = score >= 0.55
        return stable, float(score), center

# ---------------------------
# ENGINE
# ---------------------------
class Engine:
    def __init__(self):
        self.model = YOLO(MODEL_PATH)
        self.draw = Drawer()

    def process_person(self, frame, kpts, conf, person_id):
        h, w = frame.shape[:2]

        # Draw skeleton
        for a, b in SKELETON:
            if conf[a] > MIN_CONF and conf[b] > MIN_CONF:
                self.draw.line(frame, kpts[a], kpts[b], (180, 180, 180), 2)

        # Draw all keypoints
        for i in range(len(kpts)):
            if conf[i] > MIN_CONF:
                self.draw.point(frame, kpts[i], (0, 255, 255), 4)

        # Estimate COM / COG
        com = Biomech.estimate_com(kpts, conf)

        # Foot polygons and support polygon
        support_hull, support_pts = Biomech.support_polygon(kpts, conf)

        l_kn = get_point(kpts, conf, KP["l_kn"])
        l_an = get_point(kpts, conf, KP["l_an"])
        r_kn = get_point(kpts, conf, KP["r_kn"])
        r_an = get_point(kpts, conf, KP["r_an"])

        # Draw foot normals and pseudo-foot geometry
        foot_meta = {
            "left": None,
            "right": None
        }

        if l_kn is not None and l_an is not None:
            poly, center, axes = Biomech.estimate_foot_polygon(l_kn, l_an, "left")
            if poly is not None:
                foot_meta["left"] = {"poly": poly, "center": center}
                if DRAW_NORMALS:
                    foot_axis, foot_normal = axes
                    self.draw.line(frame, center - 30 * foot_axis, center + 30 * foot_axis, (0, 180, 255), 2)
                    self.draw.line(frame, center - 35 * foot_normal, center + 35 * foot_normal, (255, 120, 0), 2)
                    self.draw.label(frame, "foot normal", center + np.array([5, -5], dtype=np.float32), (255, 120, 0), 0.45, 1)

        if r_kn is not None and r_an is not None:
            poly, center, axes = Biomech.estimate_foot_polygon(r_kn, r_an, "right")
            if poly is not None:
                foot_meta["right"] = {"poly": poly, "center": center}
                if DRAW_NORMALS:
                    foot_axis, foot_normal = axes
                    self.draw.line(frame, center - 30 * foot_axis, center + 30 * foot_axis, (0, 180, 255), 2)
                    self.draw.line(frame, center - 35 * foot_normal, center + 35 * foot_normal, (255, 120, 0), 2)

        # Draw support hull
        if DRAW_SUPPORTHULL and support_hull is not None and len(support_hull) >= 2:
            hull_int = np.round(support_hull).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [hull_int], True, (0, 255, 0), 2)
            for p in support_hull:
                self.draw.point(frame, p, (0, 255, 0), 4)

        # Draw angles and biomechanics vectors
        angles = {}
        reactions = {}

        for name, (a_i, b_i, c_i) in JOINTS.items():
            if conf[a_i] > MIN_CONF and conf[b_i] > MIN_CONF and conf[c_i] > MIN_CONF:
                a = kpts[a_i]
                b = kpts[b_i]
                c = kpts[c_i]

                ang = Vec.angle(a, b, c)
                angles[name] = float(ang)

                # Angle arc
                self.draw.arc(frame, a, b, c, (0, 255, 255))
                self.draw.label(frame, f"{int(round(ang))}°", b + np.array([6, -6], dtype=np.float32), (255, 255, 255), 0.5, 1)

                if DRAW_REACTION_VECTORS:
                    vec, mag, ang2, direction, flex, com_load = Biomech.joint_reaction_vector(a, b, c, com=com, base=55.0)

                    # Color by load
                    if mag < 110:
                        color = (0, 255, 0)
                    elif mag < 170:
                        color = (0, 215, 255)
                    else:
                        color = (0, 0, 255)

                    self.draw.arrow(frame, b, direction, color, f"{int(round(mag))}", scale=0.35)

                    reactions[name] = {
                        "vector": [float(vec[0]), float(vec[1])],
                        "magnitude": float(mag),
                        "angle_deg": float(ang2),
                        "flex_ratio": float(flex),
                        "com_bias": float(com_load),
                    }

        # Draw COM / COG
        if DRAW_COM_COG and com is not None:
            self.draw.point(frame, com, (255, 0, 255), 7)
            self.draw.label(frame, "COM / COG", com + np.array([8, -8], dtype=np.float32), (255, 0, 255), 0.55, 2)

            # vertical projection line to ground/support level
            if support_pts:
                ground_y = float(np.max(np.array(support_pts, dtype=np.float32)[:, 1]))
                proj = np.array([com[0], ground_y], dtype=np.float32)
                self.draw.line(frame, com, proj, (255, 0, 255), 1)
                self.draw.point(frame, proj, (255, 0, 255), 3)

        # Stability
        stable, score, support_center = Biomech.stability_score(com, support_hull, support_pts, w)

        # Draw stability text
        stability_text = f"STABLE: {stable}   SCORE: {score:.2f}"
        stability_color = (0, 255, 0) if stable else (0, 0, 255)
        cv2.rectangle(frame, (15, 15), (420, 95), (20, 20, 20), -1)
        cv2.rectangle(frame, (15, 15), (420, 95), stability_color, 2)
        self.draw.label(frame, stability_text, np.array([28, 45], dtype=np.float32), stability_color, 0.7, 2)

        if com is not None:
            self.draw.label(frame, f"COM: ({int(com[0])}, {int(com[1])})", np.array([28, 68], dtype=np.float32), (255, 0, 255), 0.55, 1)

        if support_center is not None:
            self.draw.point(frame, support_center, (0, 255, 0), 5)
            self.draw.label(frame, "support center", support_center + np.array([6, 14], dtype=np.float32), (0, 255, 0), 0.45, 1)

        # Person label
        nose = get_point(kpts, conf, KP["nose"])
        if nose is not None:
            anchor = nose
        else:
            valid_pts = kpts[conf > MIN_CONF]
            anchor = valid_pts[0] if len(valid_pts) else np.array([30, 120 + 25 * person_id], dtype=np.float32)

        self.draw.label(frame, f"Person {person_id}", anchor + np.array([0, -18], dtype=np.float32), (0, 255, 255), 0.6, 2)

        # Prepare record
        record = {
            "person_id": int(person_id),
            "com": [float(com[0]), float(com[1])] if com is not None else None,
            "cog": [float(com[0]), float(com[1])] if com is not None else None,
            "support_hull": support_hull.tolist() if support_hull is not None else None,
            "stability": {
                "stable": bool(stable),
                "score": float(score),
            },
            "angles": angles,
            "reactions": reactions,
            "left_foot": {
                "ankle": [float(l_an[0]), float(l_an[1])] if l_an is not None else None,
                "knee": [float(l_kn[0]), float(l_kn[1])] if l_kn is not None else None,
            },
            "right_foot": {
                "ankle": [float(r_an[0]), float(r_an[1])] if r_an is not None else None,
                "knee": [float(r_kn[0]), float(r_kn[1])] if r_kn is not None else None,
            }
        }

        return frame, record

    def process_frame(self, frame):
        results = self.model(frame, verbose=False)

        frame_records = []
        if not results or len(results) == 0 or results[0].keypoints is None:
            cv2.putText(frame, "No person detected", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame, frame_records

        r = results[0]
        if r.keypoints is None:
            cv2.putText(frame, "No person detected", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame, frame_records

        # Multi-person pose
        kpts_all = r.keypoints.xy.cpu().numpy()
        conf_all = r.keypoints.conf.cpu().numpy()

        if len(kpts_all) == 0:
            cv2.putText(frame, "No person detected", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame, frame_records

        for pid, (kpts, conf) in enumerate(zip(kpts_all, conf_all)):
            record = None
            frame, record = self.process_person(frame, kpts, conf, pid)
            frame_records.append(record)

        return frame, frame_records

# ---------------------------
# MAIN
# ---------------------------
def main():
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
            "frame": int(frame_idx),
            "time": round(frame_idx / fps, 4),
            "num_persons": int(len(people)),
            "persons": people
        })

        if SHOW_LIVE:
            cv2.imshow("Biomechanics Pose Inference", annotated)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to stop
                break

        frame_idx += 1

    cap.release()
    out.release()
    if SHOW_LIVE:
        cv2.destroyAllWindows()

    with open(OUTPUT_JSON, "w") as f:
        json.dump(timeline, f, indent=2)

    print(f"Done. Saved: {OUTPUT_VIDEO}")
    print(f"Done. Saved: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
