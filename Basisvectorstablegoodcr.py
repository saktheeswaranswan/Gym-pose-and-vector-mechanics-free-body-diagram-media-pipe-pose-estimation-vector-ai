# Improved Biomechanics Pose Estimation with Stability
# Saves annotated output video only

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
        """
        Approximate foot polygon from knee->ankle direction.
        """
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

        # Head
        if pts["nose"] is not None and pts["l_sh"] is not None and pts["r_sh"] is not None:
            shoulders = 0.5 * (pts["l_sh"] + pts["r_sh"])
            head_center = 0.5 * (pts["nose"] + shoulders)
            segments.append((head_center, SEGMENT_WEIGHTS["head"]))

        # Trunk
        if pts["l_sh"] is not None and pts["r_sh"] is not None and pts["l_hp"] is not None and pts["r_hp"] is not None:
            shoulders = 0.5 * (pts["l_sh"] + pts["r_sh"])
            hips = 0.5 * (pts["l_hp"] + pts["r_hp"])
            trunk_center = 0.5 * (shoulders + hips)
            segments.append((trunk_center, SEGMENT_WEIGHTS["trunk"]))

        # Upper arms
        if pts["l_sh"] is not None and pts["l_el"] is not None:
            segments.append((0.5 * (pts["l_sh"] + pts["l_el"]), SEGMENT_WEIGHTS["upper_arm_l"]))
        if pts["r_sh"] is not None and pts["r_el"] is not None:
            segments.append((0.5 * (pts["r_sh"] + pts["r_el"]), SEGMENT_WEIGHTS["upper_arm_r"]))

        # Forearms
        if pts["l_el"] is not None and pts["l_wr"] is not None:
            segments.append((0.5 * (pts["l_el"] + pts["l_wr"]), SEGMENT_WEIGHTS["forearm_l"]))
        if pts["r_el"] is not None and pts["r_wr"] is not None:
            segments.append((0.5 * (pts["r_el"] + pts["r_wr"]), SEGMENT_WEIGHTS["forearm_r"]))

        # Thighs
        if pts["l_hp"] is not None and pts["l_kn"] is not None:
            segments.append((0.5 * (pts["l_hp"] + pts["l_kn"]), SEGMENT_WEIGHTS["thigh_l"]))
        if pts["r_hp"] is not None and pts["r_kn"] is not None:
            segments.append((0.5 * (pts["r_hp"] + pts["r_kn"]), SEGMENT_WEIGHTS["thigh_r"]))

        # Shanks
        if pts["l_kn"] is not None and pts["l_an"] is not None:
            segments.append((0.5 * (pts["l_kn"] + pts["l_an"]), SEGMENT_WEIGHTS["shank_l"]))
        if pts["r_kn"] is not None and pts["r_an"] is not None:
            segments.append((0.5 * (pts["r_kn"] + pts["r_an"]), SEGMENT_WEIGHTS["shank_r"]))

        # Feet
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

        # Left foot
        if pts["l_kn"] is not None and pts["l_an"] is not None:
            poly, _, _ = Biomech.estimate_foot_polygon(pts["l_kn"], pts["l_an"], "left")
            if poly is not None:
                support_pts.extend(poly.tolist())
        elif pts["l_an"] is not None:
            support_pts.append(pts["l_an"])

        # Right foot
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

        # Convex hull via OpenCV for stability
        hull = cv2.convexHull(arr.reshape(-1, 1, 2)).reshape(-1, 2).astype(np.float32)
        return hull, support_pts

    @staticmethod
    def joint_reaction_vector(a, b, c, com=None, base=60.0):
        """
        Heuristic reaction vector at joint b given points a-b-c.
        """
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
    def stability_score(com, hull, support_pts, frame_w):
        """
        2D stability estimate:
        - COM inside support hull is preferred
        - Larger edge margin is better
        - For 2-point support, use horizontal COM alignment
        """
        if com is None:
            return False, 0.0, None

        # Fallback: two support points only
        if hull is None or len(hull) < 3:
            if len(support_pts) >= 2:
                xs = [float(p[0]) for p in support_pts]
                xmin, xmax = min(xs), max(xs)
                width = max(20.0, xmax - xmin)
                center_x = 0.5 * (xmin + xmax)
                dist = abs(float(com[0]) - center_x)
                score = math.exp(-dist / (0.35 * width + 1e-6))
                stable = score >= 0.55
                return stable, float(score), np.array([center_x, float(np.mean([p[1] for p in support_pts]))], dtype=np.float32)

            return False, 0.0, None

        hull_pts = hull.astype(np.float32)
        center = np.mean(hull_pts, axis=0)

        inside = (cv2.pointPolygonTest(hull_pts.reshape(-1, 1, 2), (float(com[0]), float(com[1])), False) >= 0)

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

        xs = hull_pts[:, 0]
        width = max(30.0, float(np.max(xs) - np.min(xs)))
        x_dist = abs(float(com[0]) - float(center[0]))

        x_score = math.exp(-x_dist / (0.35 * width + 1e-6))
        margin_score = min(1.0, edge_dist / (0.18 * width + 1e-6))

        score = 0.58 * x_score + 0.42 * margin_score
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

        # Foot normals and axes
        if DRAW_NORMALS:
            if pts["l_kn"] is not None and pts["l_an"] is not None:
                poly, center, axes = Biomech.estimate_foot_polygon(pts["l_kn"], pts["l_an"], "left")
                if poly is not None and axes is not None:
                    foot_axis, foot_normal = axes
                    self.draw.line(frame, center - 30 * foot_axis, center + 30 * foot_axis, (0, 180, 255), 2)
                    self.draw.line(frame, center - 35 * foot_normal, center + 35 * foot_normal, (255, 120, 0), 2)
            if pts["r_kn"] is not None and pts["r_an"] is not None:
                poly, center, axes = Biomech.estimate_foot_polygon(pts["r_kn"], pts["r_an"], "right")
                if poly is not None and axes is not None:
                    foot_axis, foot_normal = axes
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

        # Stability
        stable, score, support_center = Biomech.stability_score(com, hull, support_pts, frame.shape[1])
        stability_text = f"STABLE" if stable else f"UNSTABLE"
        stability_text2 = f"SCORE: {score:.2f}"

        box_color = (0, 200, 0) if stable else (0, 0, 255)
        cv2.rectangle(frame, (15, 15), (430, 105), (20, 20, 20), -1)
        cv2.rectangle(frame, (15, 15), (430, 105), box_color, 2)

        self.draw.label(frame, stability_text, (28, 45), box_color, 0.8, 2)
        self.draw.label(frame, stability_text2, (28, 75), (255, 255, 255), 0.6, 2)

        if com is not None and DRAW_COM_COG:
            self.draw.point(frame, com, (255, 0, 255), 7)
            self.draw.label(frame, "COM/COG", (com[0] + 8, com[1] - 8), (255, 0, 255), 0.55, 2)

            if support_pts:
                ground_y = float(max(p[1] for p in support_pts))
                proj = np.array([com[0], ground_y], dtype=np.float32)
                self.draw.line(frame, com, proj, (255, 0, 255), 1)
                self.draw.point(frame, proj, (255, 0, 255), 3)

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
            "support_hull": hull.tolist() if hull is not None else None,
            "stability": {"stable": bool(stable), "score": float(score)},
            "angles": angles,
            "reactions": reactions,
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
