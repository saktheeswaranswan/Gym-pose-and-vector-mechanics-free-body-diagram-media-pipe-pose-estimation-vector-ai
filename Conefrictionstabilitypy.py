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

DRAW_SUPPORTHULL = True
DRAW_NORMALS = True
DRAW_COM_COG = True
DRAW_REACTION_VECTORS = True
DRAW_JOINT_ANGLES = True
DRAW_PERSON_LABEL = True

# NEW: ground contact / friction model
MASS_KG = 70.0
GRAVITY = 9.81
MU_FRICTION = 0.65
DRAW_FRICTION_CONE = True
DRAW_PRESSURE_CENTER = True
DRAW_GROUND_REACTION_FORCES = True

# ---------------------------
# COCO KEYPOINTS
# ---------------------------
KP = {
    "nose": 0, "l_eye": 1, "r_eye": 2, "l_ear": 3, "r_ear": 4,
    "l_sh": 5, "r_sh": 6, "l_el": 7, "r_el": 8, "l_wr": 9, "r_wr": 10,
    "l_hp": 11, "r_hp": 12, "l_kn": 13, "r_kn": 14, "l_an": 15, "r_an": 16
}

SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16)
]

JOINTS = {
    "left_elbow": (KP["l_sh"], KP["l_el"], KP["l_wr"]),
    "right_elbow": (KP["r_sh"], KP["r_el"], KP["r_wr"]),
    "left_knee": (KP["l_hp"], KP["l_kn"], KP["l_an"]),
    "right_knee": (KP["r_hp"], KP["r_kn"], KP["r_an"]),
    "left_hip": (KP["l_sh"], KP["l_hp"], KP["l_kn"]),
    "right_hip": (KP["r_sh"], KP["r_hp"], KP["r_kn"]),
    "left_shoulder": (KP["l_el"], KP["l_sh"], KP["l_hp"]),
    "right_shoulder": (KP["r_el"], KP["r_sh"], KP["r_hp"]),
}

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
# VECTOR HELPERS
# ---------------------------
class Vec:
    @staticmethod
    def norm(v):
        return float(np.linalg.norm(v))

    @staticmethod
    def unit(v):
        n = np.linalg.norm(v)
        return (v / n) if n > 1e-8 else np.zeros_like(v, dtype=np.float32)

    @staticmethod
    def perp(v):
        return np.array([-v[1], v[0]], dtype=np.float32)

    @staticmethod
    def rotate(v, theta):
        c, s = math.cos(theta), math.sin(theta)
        return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]], dtype=np.float32)

    @staticmethod
    def mid(a, b):
        return (a + b) * 0.5

    @staticmethod
    def angle(a, b, c):
        ba = Vec.unit(a - b)
        bc = Vec.unit(c - b)
        val = np.clip(np.dot(ba, bc), -1.0, 1.0)
        return float(np.degrees(np.arccos(val)))

# ---------------------------
# DRAWING HELPERS
# ---------------------------
class Drawer:
    def point(self, frame, p, color=(0, 255, 255), r=4):
        cv2.circle(frame, (int(p[0]), int(p[1])), r, color, -1)

    def line(self, frame, p1, p2, color=(200, 200, 200), thickness=2):
        cv2.line(
            frame,
            (int(p1[0]), int(p1[1])),
            (int(p2[0]), int(p2[1])),
            color,
            thickness,
            cv2.LINE_AA,
        )

    def arrow(self, frame, p, v, color=(255, 0, 0), text=None, scale=1.0):
        end = p + v * scale
        cv2.arrowedLine(
            frame,
            (int(p[0]), int(p[1])),
            (int(end[0]), int(end[1])),
            color,
            2,
            cv2.LINE_AA,
            tipLength=0.22,
        )
        if text is not None:
            self.label(frame, text, (end[0] + 6, end[1] - 4), color, 0.45, 1)

    def arc(self, frame, a, b, c, color=(0, 255, 255), radius=22):
        ang1 = math.atan2(a[1] - b[1], a[0] - b[0])
        ang2 = math.atan2(c[1] - b[1], c[0] - b[0])
        diff = (ang2 - ang1 + math.pi) % (2 * math.pi) - math.pi

        pts = []
        for t in np.linspace(0.0, 1.0, 20):
            ang = ang1 + diff * t
            x = int(b[0] + radius * math.cos(ang))
            y = int(b[1] + radius * math.sin(ang))
            pts.append([x, y])

        pts = np.array(pts, np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], False, color, 2, cv2.LINE_AA)

    def label(self, frame, text, p, color=(255, 255, 255), scale=0.55, thickness=1):
        x, y = int(p[0]), int(p[1])
        cv2.putText(
            frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
            scale, color, thickness, cv2.LINE_AA
        )

    def label_bg(self, frame, text, p, text_color=(255, 255, 255),
                 bg_color=(20, 20, 20), scale=0.6, thickness=2, pad=6):
        x, y = int(p[0]), int(p[1])
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        cv2.rectangle(frame, (x - pad, y - th - pad), (x + tw + pad, y + baseline + pad), bg_color, -1)
        cv2.putText(
            frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
            scale, text_color, thickness, cv2.LINE_AA
        )

# ---------------------------
# BIOMECHANICS HELPERS
# ---------------------------
class Biomech:
    @staticmethod
    def estimate_foot_polygon(knee, ankle, side="left"):
        shank = ankle - knee
        L = Vec.norm(shank)
        if L < 1e-6:
            return None, None, None

        shank_u = Vec.unit(shank)
        normal = Vec.perp(shank_u)
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
        foot_axis = Vec.unit(toe - heel)
        foot_normal = Vec.perp(foot_axis)

        return poly, center, (foot_axis, foot_normal)

    @staticmethod
    def estimate_com(kpts, conf):
        pts = {
            name: (np.array(kpts[idx], dtype=np.float32) if conf[idx] > MIN_CONF else None)
            for name, idx in KP.items()
        }

        segments = []

        if pts["nose"] is not None and pts["l_sh"] is not None and pts["r_sh"] is not None:
            shoulders = 0.5 * (pts["l_sh"] + pts["r_sh"])
            head_center = 0.5 * (pts["nose"] + shoulders)
            segments.append((head_center, SEGMENT_WEIGHTS["head"]))

        if pts["l_sh"] is not None and pts["r_sh"] is not None and pts["l_hp"] is not None and pts["r_hp"] is not None:
            shoulders = 0.5 * (pts["l_sh"] + pts["r_sh"])
            hips = 0.5 * (pts["l_hp"] + pts["r_hp"])
            trunk_center = 0.5 * (shoulders + hips)
            segments.append((trunk_center, SEGMENT_WEIGHTS["trunk"]))

        if pts["l_sh"] is not None and pts["l_el"] is not None:
            segments.append((0.5 * (pts["l_sh"] + pts["l_el"]), SEGMENT_WEIGHTS["upper_arm_l"]))
        if pts["r_sh"] is not None and pts["r_el"] is not None:
            segments.append((0.5 * (pts["r_sh"] + pts["r_el"]), SEGMENT_WEIGHTS["upper_arm_r"]))

        if pts["l_el"] is not None and pts["l_wr"] is not None:
            segments.append((0.5 * (pts["l_el"] + pts["l_wr"]), SEGMENT_WEIGHTS["forearm_l"]))
        if pts["r_el"] is not None and pts["r_wr"] is not None:
            segments.append((0.5 * (pts["r_el"] + pts["r_wr"]), SEGMENT_WEIGHTS["forearm_r"]))

        if pts["l_hp"] is not None and pts["l_kn"] is not None:
            segments.append((0.5 * (pts["l_hp"] + pts["l_kn"]), SEGMENT_WEIGHTS["thigh_l"]))
        if pts["r_hp"] is not None and pts["r_kn"] is not None:
            segments.append((0.5 * (pts["r_hp"] + pts["r_kn"]), SEGMENT_WEIGHTS["thigh_r"]))

        if pts["l_kn"] is not None and pts["l_an"] is not None:
            segments.append((0.5 * (pts["l_kn"] + pts["l_an"]), SEGMENT_WEIGHTS["shank_l"]))
        if pts["r_kn"] is not None and pts["r_an"] is not None:
            segments.append((0.5 * (pts["r_kn"] + pts["r_an"]), SEGMENT_WEIGHTS["shank_r"]))

        if pts["l_kn"] is not None and pts["l_an"] is not None:
            poly, center, _ = Biomech.estimate_foot_polygon(pts["l_kn"], pts["l_an"], "left")
            if center is not None:
                segments.append((center, SEGMENT_WEIGHTS["foot_l"]))
        if pts["r_kn"] is not None and pts["r_an"] is not None:
            poly, center, _ = Biomech.estimate_foot_polygon(pts["r_kn"], pts["r_an"], "right")
            if center is not None:
                segments.append((center, SEGMENT_WEIGHTS["foot_r"]))

        if not segments:
            return None

        arr = np.array([seg[0] for seg in segments], dtype=np.float32)
        ws = np.array([seg[1] for seg in segments], dtype=np.float32)
        ws = ws / (np.sum(ws) + 1e-8)
        com = np.sum(arr * ws[:, None], axis=0)
        return com

    @staticmethod
    def support_polygon(kpts, conf):
        pts = {
            name: (np.array(kpts[idx], dtype=np.float32) if conf[idx] > MIN_CONF else None)
            for name, idx in KP.items()
        }

        support_pts = []

        if pts["l_kn"] is not None and pts["l_an"] is not None:
            poly, _, _ = Biomech.estimate_foot_polygon(pts["l_kn"], pts["l_an"], "left")
            if poly is not None:
                support_pts.extend(poly.tolist())
        elif pts["l_an"] is not None:
            support_pts.append(pts["l_an"])

        if pts["r_kn"] is not None and pts["r_an"] is not None:
            poly, _, _ = Biomech.estimate_foot_polygon(pts["r_kn"], pts["r_an"], "right")
            if poly is not None:
                support_pts.extend(poly.tolist())
        elif pts["r_an"] is not None:
            support_pts.append(pts["r_an"])

        if len(support_pts) < 2:
            return None, support_pts

        arr = np.array(support_pts, dtype=np.float32)
        if len(arr) == 2:
            return arr, support_pts

        hull = cv2.convexHull(arr.reshape(-1, 1, 2)).reshape(-1, 2).astype(np.float32)
        return hull, support_pts

    @staticmethod
    def joint_reaction_vector(a, b, c, com=None, base=60.0):
        u1 = Vec.unit(a - b)
        u2 = Vec.unit(c - b)
        ang = Vec.angle(a, b, c)
        flex = abs(180.0 - ang) / 180.0
        align = abs(np.dot(u1, u2))

        bisector = u1 + u2
        if Vec.norm(bisector) < 1e-6:
            bisector = Vec.perp(u2)
        bisector = Vec.unit(bisector)

        distal_normal = Vec.perp(u2)
        gravity = np.array([0.0, -1.0], dtype=np.float32)

        com_bias = np.zeros(2, dtype=np.float32)
        if com is not None and Vec.norm(com - b) > 1e-6:
            com_bias = Vec.unit(com - b)

        direction = Vec.unit(
            0.38 * bisector +
            0.28 * distal_normal +
            0.18 * gravity +
            0.16 * com_bias
        )

        if Vec.norm(direction) < 1e-6:
            direction = Vec.unit(distal_normal if Vec.norm(distal_normal) > 1e-6 else gravity)

        com_load = 0.0
        if com is not None and Vec.norm(com - b) > 1e-6:
            com_dir = Vec.unit(com - b)
            com_load = abs(np.dot(com_dir, distal_normal))

        mag = base + 140.0 * flex + 35.0 * (1.0 - align) + 55.0 * com_load
        return direction * mag, float(mag), float(ang), direction

    @staticmethod
    def ground_reaction_model(com, hull, support_pts, left_foot, right_foot,
                              mass_kg=70.0, mu=0.65):
        """
        Estimated ground contact model from RGB pose only.
        Returns:
            stable, score, support_center, cop, foot_forces, slip_ratio
        """
        if com is None or not support_pts:
            return False, 0.0, None, None, [], 1.0

        weight = float(mass_kg * GRAVITY)
        feet = []

        for foot in (left_foot, right_foot):
            if not foot:
                continue
            center = foot.get("center", None)
            axis = foot.get("axis", None)
            normal = foot.get("normal", None)
            if center is None or axis is None or normal is None:
                continue
            feet.append({
                "center": np.array(center, dtype=np.float32),
                "axis": Vec.unit(np.array(axis, dtype=np.float32)),
                "normal": Vec.unit(np.array(normal, dtype=np.float32)),
            })

        if len(feet) == 0:
            return False, 0.0, None, None, [], 1.0

        hull_pts = None
        inside = False
        if hull is not None and len(hull) >= 3:
            hull_pts = hull.astype(np.float32)
            support_center = np.mean(hull_pts, axis=0)
            xmin = float(np.min(hull_pts[:, 0]))
            xmax = float(np.max(hull_pts[:, 0]))
            ymax = float(np.max(hull_pts[:, 1]))

            cop = np.array([float(np.clip(com[0], xmin, xmax)), ymax], dtype=np.float32)
            inside = cv2.pointPolygonTest(
                hull_pts.reshape(-1, 1, 2),
                (float(com[0]), float(com[1])),
                False
            ) >= 0
        else:
            xs = np.array([p[0] for p in support_pts], dtype=np.float32)
            ys = np.array([p[1] for p in support_pts], dtype=np.float32)
            support_center = np.array([float(np.mean(xs)), float(np.mean(ys))], dtype=np.float32)
            cop = np.array([float(np.clip(com[0], float(np.min(xs)), float(np.max(xs)))),
                            float(np.max(ys))], dtype=np.float32)
            inside = len(support_pts) >= 2

        # Center of pressure on ground-support line
        if DRAW_PRESSURE_CENTER and cop is None:
            cop = support_center

        # Total friction demand from COM offset about the support center
        dx = float(com[0] - cop[0])
        dy = max(20.0, float(abs(com[1] - cop[1])))

        lean_angle = math.atan2(abs(dx), dy)
        traction_demand_total = weight * math.tan(lean_angle)
        traction_limit_total = mu * weight
        slip_ratio = float(traction_demand_total / (traction_limit_total + 1e-8))

        # Foot load split based on distance to COP
        dists = np.array([np.linalg.norm(foot["center"] - cop) for foot in feet], dtype=np.float32)
        inv = 1.0 / (dists + 1e-6)
        load_weights = inv / (np.sum(inv) + 1e-8)

        display_force_scale = 0.08
        foot_forces = []
        for foot, lw in zip(feet, load_weights):
            Fn = float(weight * lw)
            Ft_req = float(traction_demand_total * lw)
            Ft = float(min(Ft_req, mu * Fn))

            tangent = Vec.unit(foot["axis"])
            if Vec.norm(tangent) < 1e-6:
                tangent = np.array([1.0, 0.0], dtype=np.float32)

            # Traction opposes slip tendency along the foot axis
            slip_sign = 1.0 if dx >= 0 else -1.0
            traction_dir = -slip_sign * tangent
            normal_dir = np.array([0.0, -1.0], dtype=np.float32)

            normal_vec_px = normal_dir * Fn * display_force_scale
            traction_vec_px = traction_dir * Ft * display_force_scale

            # Friction cone boundary in 2D: angle = atan(mu)
            cone_angle = math.atan(mu)
            cone_left = Vec.rotate(normal_dir, -cone_angle)
            cone_right = Vec.rotate(normal_dir, cone_angle)

            foot_forces.append({
                "center": foot["center"].tolist(),
                "Fn": Fn,
                "Ft": Ft,
                "Ft_req": Ft_req,
                "normal_vec": normal_vec_px.tolist(),
                "traction_vec": traction_vec_px.tolist(),
                "cone_left": (cone_left * 50.0).tolist(),
                "cone_right": (cone_right * 50.0).tolist(),
                "mu": float(mu),
            })

        # Score: inside support + margin + friction utilization
        if hull_pts is not None and len(hull_pts) >= 3:
            def dist_edge(p, a, b):
                ap = p - a
                ab = b - a
                denom = np.dot(ab, ab) + 1e-8
                t = np.clip(np.dot(ap, ab) / denom, 0.0, 1.0)
                proj = a + t * ab
                return np.linalg.norm(p - proj)

            edge_dist = min(
                dist_edge(com, hull_pts[i], hull_pts[(i + 1) % len(hull_pts)])
                for i in range(len(hull_pts))
            )
            width = max(30.0, float(np.max(hull_pts[:, 0]) - np.min(hull_pts[:, 0])))
            x_dist = abs(float(com[0]) - float(support_center[0]))
            x_score = math.exp(-x_dist / (0.35 * width + 1e-6))
            margin_score = min(1.0, edge_dist / (0.18 * width + 1e-6))
        else:
            width = max(30.0, float(abs(feet[0]["center"][0] - feet[-1]["center"][0])) if len(feet) > 1 else 40.0)
            x_dist = abs(float(com[0]) - float(support_center[0]))
            x_score = math.exp(-x_dist / (0.35 * width + 1e-6))
            margin_score = 0.5

        friction_score = max(0.0, 1.0 - slip_ratio)
        inside_score = 1.0 if inside else 0.2

        score = 0.35 * inside_score + 0.30 * margin_score + 0.35 * friction_score
        stable = bool(score >= 0.55 and slip_ratio <= 1.0 and inside)

        return stable, float(score), support_center, cop, foot_forces, float(slip_ratio)

# ---------------------------
# ENGINE
# ---------------------------
class Engine:
    def __init__(self):
        self.model = YOLO(MODEL_PATH)
        self.draw = Drawer()

    def _get_anchor(self, kpts):
        if kpts is None or len(kpts) == 0:
            return np.array([30, 60], dtype=np.float32)
        valid_pts = [np.array(p, dtype=np.float32) for p in kpts if p is not None]
        return valid_pts[0] if valid_pts else np.array([30, 60], dtype=np.float32)

    def process_person(self, frame, kpts, conf, pid):
        pts = {
            name: (np.array(kpts[idx], dtype=np.float32) if conf[idx] > MIN_CONF else None)
            for name, idx in KP.items()
        }

        # Skeleton
        for a, b in SKELETON:
            if conf[a] > MIN_CONF and conf[b] > MIN_CONF:
                self.draw.line(frame, kpts[a], kpts[b], (180, 180, 180), 2)

        # Keypoints
        for i, pt in enumerate(kpts):
            if conf[i] > MIN_CONF:
                self.draw.point(frame, pt, (0, 255, 255), 4)

        # COM
        com = Biomech.estimate_com(kpts, conf)

        # Support polygon
        hull, support_pts = Biomech.support_polygon(kpts, conf)

        # Foot normals / foot axes
        left_foot = None
        right_foot = None

        if DRAW_NORMALS:
            if pts["l_kn"] is not None and pts["l_an"] is not None:
                poly, center, axes = Biomech.estimate_foot_polygon(pts["l_kn"], pts["l_an"], "left")
                if poly is not None and axes is not None:
                    foot_axis, foot_normal = axes
                    left_foot = {
                        "poly": poly.tolist(),
                        "center": center.tolist(),
                        "axis": foot_axis.tolist(),
                        "normal": foot_normal.tolist(),
                    }
                    self.draw.line(frame, center - 30 * foot_axis, center + 30 * foot_axis, (0, 180, 255), 2)
                    self.draw.line(frame, center - 35 * foot_normal, center + 35 * foot_normal, (255, 120, 0), 2)

            if pts["r_kn"] is not None and pts["r_an"] is not None:
                poly, center, axes = Biomech.estimate_foot_polygon(pts["r_kn"], pts["r_an"], "right")
                if poly is not None and axes is not None:
                    foot_axis, foot_normal = axes
                    right_foot = {
                        "poly": poly.tolist(),
                        "center": center.tolist(),
                        "axis": foot_axis.tolist(),
                        "normal": foot_normal.tolist(),
                    }
                    self.draw.line(frame, center - 30 * foot_axis, center + 30 * foot_axis, (0, 180, 255), 2)
                    self.draw.line(frame, center - 35 * foot_normal, center + 35 * foot_normal, (255, 120, 0), 2)

        # Support hull
        if DRAW_SUPPORTHULL and hull is not None and len(hull) >= 2:
            hull_i = np.round(hull).astype(np.int32).reshape(-1, 1, 2)
            if len(hull) >= 3:
                cv2.polylines(frame, [hull_i], True, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.line(
                    frame,
                    tuple(hull_i[0, 0]),
                    tuple(hull_i[1, 0]),
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )
            for p in hull:
                self.draw.point(frame, p, (0, 255, 0), 4)

        # Angles and reactions
        angles = {}
        reactions = {}

        for name, (ai, bi, ci) in JOINTS.items():
            if conf[ai] > MIN_CONF and conf[bi] > MIN_CONF and conf[ci] > MIN_CONF:
                a, b, c = kpts[ai], kpts[bi], kpts[ci]
                ang = Vec.angle(a, b, c)
                angles[name] = ang

                if DRAW_JOINT_ANGLES:
                    self.draw.arc(frame, a, b, c, (0, 255, 255), radius=22)
                    self.draw.label(frame, f"{int(round(ang))}°", (b[0] + 6, b[1] - 6), (255, 255, 255), 0.5, 1)

                if DRAW_REACTION_VECTORS:
                    vec, mag, _, direction = Biomech.joint_reaction_vector(a, b, c, com=com)
                    color = (0, 255, 0) if mag < 110 else ((0, 215, 255) if mag < 170 else (0, 0, 255))
                    self.draw.arrow(frame, b, direction, color, text=str(int(round(mag))), scale=0.35)
                    reactions[name] = {
                        "vector": [float(vec[0]), float(vec[1])],
                        "magnitude": float(mag),
                        "angle_deg": float(ang),
                    }

        # NEW: Ground reaction + friction cone + pressure center
        stable, score, support_center, cop, foot_forces, slip_ratio = Biomech.ground_reaction_model(
            com=com,
            hull=hull,
            support_pts=support_pts,
            left_foot=left_foot,
            right_foot=right_foot,
            mass_kg=MASS_KG,
            mu=MU_FRICTION,
        )

        stability_text = "STABLE" if stable else "UNSTABLE"
        stability_text2 = f"SCORE: {score:.2f}  SLIP: {slip_ratio:.2f}  MU: {MU_FRICTION:.2f}"

        box_color = (0, 200, 0) if stable else (0, 0, 255)
        cv2.rectangle(frame, (15, 15), (500, 105), (20, 20, 20), -1)
        cv2.rectangle(frame, (15, 15), (500, 105), box_color, 2)

        self.draw.label(frame, stability_text, (28, 45), box_color, 0.8, 2)
        self.draw.label(frame, stability_text2, (28, 75), (255, 255, 255), 0.55, 2)

        # COM / COG
        if com is not None and DRAW_COM_COG:
            self.draw.point(frame, com, (255, 0, 255), 7)
            self.draw.label(frame, "COM/COG", (com[0] + 8, com[1] - 8), (255, 0, 255), 0.55, 2)

            if support_pts:
                ground_y = float(max(p[1] for p in support_pts))
                proj = np.array([com[0], ground_y], dtype=np.float32)
                self.draw.line(frame, com, proj, (255, 0, 255), 1)
                self.draw.point(frame, proj, (255, 0, 255), 3)

        # Pressure center / center of pressure
        if cop is not None and DRAW_PRESSURE_CENTER:
            self.draw.point(frame, cop, (0, 255, 255), 7)
            self.draw.label(frame, "COP", (cop[0] + 8, cop[1] + 14), (0, 255, 255), 0.55, 2)

        # Foot-level normal and traction vectors
        if DRAW_GROUND_REACTION_FORCES and foot_forces:
            for f in foot_forces:
                c = np.array(f["center"], dtype=np.float32)

                normal_vec = np.array(f["normal_vec"], dtype=np.float32)
                traction_vec = np.array(f["traction_vec"], dtype=np.float32)

                self.draw.arrow(frame, c, normal_vec, (255, 255, 0), text=f'N {int(round(f["Fn"]))}', scale=1.0)
                self.draw.arrow(frame, c, traction_vec, (0, 165, 255), text=f'Ft {int(round(f["Ft"]))}', scale=1.0)

                if DRAW_FRICTION_CONE:
                    n = np.array([0.0, -1.0], dtype=np.float32)
                    cone_angle = math.atan(MU_FRICTION)
                    left_ray = Vec.rotate(n, -cone_angle) * 45.0
                    right_ray = Vec.rotate(n, cone_angle) * 45.0
                    self.draw.line(frame, c, c + left_ray, (180, 180, 0), 1)
                    self.draw.line(frame, c, c + right_ray, (180, 180, 0), 1)

        if support_center is not None:
            self.draw.point(frame, support_center, (0, 255, 0), 5)
            self.draw.label(frame, "support center", (support_center[0] + 6, support_center[1] + 14), (0, 255, 0), 0.45, 1)

        # Person label
        if DRAW_PERSON_LABEL:
            anchor = pts["nose"] if pts["nose"] is not None else self._get_anchor(kpts)
            self.draw.label(frame, f"Person {pid}", (anchor[0], anchor[1] - 18), (0, 255, 255), 0.6, 2)

        record = {
            "person_id": pid,
            "com": [float(com[0]), float(com[1])] if com is not None else None,
            "cog": [float(com[0]), float(com[1])] if com is not None else None,
            "cop": [float(cop[0]), float(cop[1])] if cop is not None else None,
            "support_hull": hull.tolist() if hull is not None else None,
            "stability": {"stable": bool(stable), "score": float(score), "slip_ratio": float(slip_ratio)},
            "mass_kg": float(MASS_KG),
            "mu": float(MU_FRICTION),
            "angles": angles,
            "reactions": reactions,
            "ground_reactions": foot_forces,
            "left_foot": {
                "ankle": [float(pts["l_an"][0]), float(pts["l_an"][1])] if pts["l_an"] is not None else None,
                "knee": [float(pts["l_kn"][0]), float(pts["l_kn"][1])] if pts["l_kn"] is not None else None,
            },
            "right_foot": {
                "ankle": [float(pts["r_an"][0]), float(pts["r_an"][1])] if pts["r_an"] is not None else None,
                "knee": [float(pts["r_kn"][0]), float(pts["r_kn"][1])] if pts["r_kn"] is not None else None,
            },
        }

        return frame, record

    def process_frame(self, frame):
        results = self.model(frame, verbose=False)
        frame_records = []

        if not results or len(results) == 0 or results[0].keypoints is None:
            cv2.putText(frame, "No person detected", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            return frame, frame_records

        kpts_all = results[0].keypoints.xy.cpu().numpy()
        conf_all = results[0].keypoints.conf.cpu().numpy()

        for pid, (kpts, conf) in enumerate(zip(kpts_all, conf_all)):
            frame, rec = self.process_person(frame, kpts, conf, pid)
            frame_records.append(rec)

        return frame, frame_records

# ---------------------------
# MAIN
# ---------------------------
def main():
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {INPUT_VIDEO}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    out = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    engine = Engine()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, _ = engine.process_frame(frame)
        out.write(annotated)

        if SHOW_LIVE:
            cv2.imshow("Biomechanics Pose Inference", annotated)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    out.release()
    if SHOW_LIVE:
        cv2.destroyAllWindows()

    print("Done. Saved:", OUTPUT_VIDEO)

if __name__ == "__main__":
    main()
