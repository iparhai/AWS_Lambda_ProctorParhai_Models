"""
MediaPipe iris tracker used by both local testing and Lambda handler.
"""

import math
import os
from collections import deque
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker
from mediapipe.tasks.python.vision import FaceLandmarkerOptions
from mediapipe.tasks.python.vision import RunningMode


LEFT_IRIS_CENTER = 468
LEFT_IRIS_RING = [469, 470, 471, 472]
RIGHT_IRIS_CENTER = 473
RIGHT_IRIS_RING = [474, 475, 476, 477]

LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145

RIGHT_EYE_OUTER = 362
RIGHT_EYE_INNER = 263
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109,
]

H_OFFSET = 0.08
V_OFFSET_UP = 0.025
V_OFFSET_DOWN = 0.045
HYSTERESIS = 0.012
H_LEFT = 0.42
H_RIGHT = 0.58
V_UP = 0.42
V_DOWN = 0.58
SMOOTH_WINDOW = 3
MIN_EYE_HEIGHT_PX = 6
MIN_EYE_WIDTH_PX = 10
MAX_EYE_RATIO_DELTA = 0.35
DEPTH_PROFILE_THRESHOLD = 0.025
HEAD_POSE_YAW_GAIN = 0.06
HEAD_POSE_PITCH_GAIN = 0.08
GAZE_CENTER_STRENGTH_THRESHOLD = 0.09
GAZE_OFFSET_X = 0.08
GAZE_OFFSET_Y = 0.0
VERTICAL_HORIZONTAL_GUARD = 0.16
CENTER_H_BAND = 0.03
CENTER_V_BAND = 0.05
GAZE_DY_UP_THRESHOLD = 0.11
GAZE_DY_DOWN_THRESHOLD = -0.11


def _resolve_model_path() -> str:
    module_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(module_dir))
    candidates = [
        os.path.join(module_dir, "face_landmarker.task"),
        os.path.join(project_root, "face_landmarker.task"),
        os.path.join(os.getcwd(), "face_landmarker.task"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "face_landmarker.task not found. Place it in project root or src/gaze_tracking/."
    )


class IrisTracker:
    def __init__(
        self,
        min_face_detection_confidence: float = 0.7,
        min_face_presence_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7,
    ):
        model_path = _resolve_model_path()
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._ts_ms = 0

        self._calib_samples = []
        self._calib_needed = 30
        self._calib_done = False
        self._center_h = 0.5
        self._center_v = 0.5
        self._h_history = deque(maxlen=SMOOTH_WINDOW)
        self._v_history = deque(maxlen=SMOOTH_WINDOW)
        self._last_direction = "CENTER"

    @staticmethod
    def _lm_px(lm, w: int, h: int):
        return int(lm.x * w), int(lm.y * h)

    @staticmethod
    def _iris_center_px(lm, center_idx: int, ring_indices, w: int, h: int):
        center_x = lm[center_idx].x * w
        center_y = lm[center_idx].y * h
        ring_xs = [lm[i].x * w for i in ring_indices]
        ring_ys = [lm[i].y * h for i in ring_indices]
        if ring_xs and ring_ys:
            # Blend the explicit iris center with the iris contour for stability.
            cx = 0.7 * center_x + 0.3 * float(np.mean(ring_xs))
            cy = 0.7 * center_y + 0.3 * float(np.mean(ring_ys))
        else:
            cx, cy = center_x, center_y
        return int(round(cx)), int(round(cy))

    @staticmethod
    def _avg_depth(lm, outer_idx: int, inner_idx: int) -> float:
        return float((lm[outer_idx].z + lm[inner_idx].z) / 2.0)

    @staticmethod
    def _eye_opening(outer, inner, top, bottom) -> tuple[float, float, float]:
        width = float(abs(inner[0] - outer[0]))
        height = float(abs(bottom[1] - top[1]))
        quality = width * height
        return width, height, quality

    def _gaze_vector(self, iris_cx, iris_cy, outer, inner, top, bottom, side: str):
        eye_center = [
            (outer[0] + inner[0]) / 2.0,
            (top[1] + bottom[1]) / 2.0,
        ]
        eye_width, eye_height, _ = self._eye_opening(outer, inner, top, bottom)
        if eye_width <= 1e-6 or eye_height <= 1e-6:
            return 0.0, 0.0

        eye_diff_x = (eye_center[0] - iris_cx) / eye_width - GAZE_OFFSET_X
        eye_diff_y = (iris_cy - eye_center[1]) / eye_height - GAZE_OFFSET_Y

        # Mirror the horizontal component for the right eye so both eyes share the same orientation.
        if side == "right":
            eye_diff_x *= -1.0

        bearing = math.atan2(eye_diff_y, eye_diff_x)
        bearing = (bearing + (math.pi / 2.0)) % math.pi
        strength = float(math.sqrt((eye_diff_x * eye_diff_x) + (eye_diff_y * eye_diff_y)))
        return bearing, strength, float(eye_diff_x), float(eye_diff_y)

    @staticmethod
    def _ratios(iris_cx, iris_cy, outer, inner, top, bottom):
        # Keep geometric direction (outer -> inner) per eye.
        # This avoids right-eye inversion and 0/1 saturation artifacts.
        eye_w_signed = (inner[0] - outer[0])
        eye_h = abs(bottom[1] - top[1])
        x0 = outer[0]
        y0 = min(top[1], bottom[1])
        h_r = (iris_cx - x0) / eye_w_signed if abs(eye_w_signed) > 1e-6 else 0.5
        v_r = (iris_cy - y0) / eye_h if eye_h else 0.5
        # Clamp to physical range; noisy landmarks can otherwise produce negatives or >1.
        h_r = max(0.0, min(1.0, float(h_r)))
        v_r = max(0.0, min(1.0, float(v_r)))
        return h_r, v_r

    @staticmethod
    def _face_bbox(lm, w: int, h: int):
        pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in FACE_OVAL]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)
        return [x1, y1, x2 - x1, y2 - y1]

    @staticmethod
    def _estimate_head_pose_offsets(lm, w: int, h: int, face_bbox):
        """
        Lightweight head-pose proxy from 2D landmarks.
        Returns normalized yaw/pitch offsets in roughly [-1, 1].
        """
        x, y, bw, bh = face_bbox
        if bw <= 0 or bh <= 0:
            return 0.0, 0.0

        nose = (lm[1].x * w, lm[1].y * h)
        face_cx = x + (bw / 2.0)
        face_cy = y + (bh / 2.0)

        yaw = (nose[0] - face_cx) / max(1.0, bw / 2.0)
        pitch = (nose[1] - face_cy) / max(1.0, bh / 2.0)

        yaw = max(-1.0, min(1.0, float(yaw)))
        pitch = max(-1.0, min(1.0, float(pitch)))
        return yaw, pitch

    def _classify(
        self,
        h: float,
        v: float,
        center_h: Optional[float],
        center_v: Optional[float],
        gaze_strength: float = 1.0,
        gaze_dy: float = 0.0,
    ) -> str:
        if center_h is not None and center_v is not None:
            base_h, base_v = center_h, center_v
        elif self._calib_done:
            base_h, base_v = self._center_h, self._center_v
        else:
            base_h, base_v = 0.5, 0.5

        up_enter = base_v - V_OFFSET_UP
        down_enter = base_v + V_OFFSET_DOWN
        left_enter = base_h - H_OFFSET
        right_enter = base_h + H_OFFSET

        up_stay = up_enter + HYSTERESIS
        down_stay = down_enter - HYSTERESIS
        left_stay = left_enter + HYSTERESIS
        right_stay = right_enter - HYSTERESIS

        if self._last_direction == "UP" and v < up_stay and gaze_dy >= (GAZE_DY_UP_THRESHOLD * 0.9):
            return "UP"
        if self._last_direction == "DOWN" and v > down_stay and gaze_dy <= (GAZE_DY_DOWN_THRESHOLD * 0.9):
            return "DOWN"
        if self._last_direction == "LEFT" and h < left_stay:
            return "LEFT"
        if self._last_direction == "RIGHT" and h > right_stay:
            return "RIGHT"

        vertical_delta = base_v - v
        horizontal_delta = h - base_h

        if abs(horizontal_delta) < CENTER_H_BAND and abs(vertical_delta) < CENTER_V_BAND and abs(gaze_dy) < 0.08:
            return "CENTER"

        # Vertical gaze is usually weaker than horizontal on webcam input.
        # Let vertical displacement win if it is clearly above the calibrated baseline.
        if gaze_dy >= GAZE_DY_UP_THRESHOLD and vertical_delta > 0.015 and abs(horizontal_delta) < VERTICAL_HORIZONTAL_GUARD:
            return "UP"
        if vertical_delta > max(V_OFFSET_UP, 0.05) and abs(horizontal_delta) < VERTICAL_HORIZONTAL_GUARD:
            return "UP"
        if gaze_dy <= GAZE_DY_DOWN_THRESHOLD and vertical_delta < -0.015 and abs(horizontal_delta) < VERTICAL_HORIZONTAL_GUARD:
            return "DOWN"
        if vertical_delta < -max(V_OFFSET_DOWN, 0.05) and abs(horizontal_delta) < VERTICAL_HORIZONTAL_GUARD:
            return "DOWN"

        if gaze_strength < GAZE_CENTER_STRENGTH_THRESHOLD:
            return "CENTER"

        if v < up_enter:
            return "UP"
        if v > down_enter:
            return "DOWN"
        if h < left_enter:
            return "LEFT"
        if h > right_enter:
            return "RIGHT"

        if center_h is None and center_v is None and not self._calib_done:
            if v < V_UP:
                return "UP"
            if v > V_DOWN:
                return "DOWN"
            if h < H_LEFT:
                return "LEFT"
            if h > H_RIGHT:
                return "RIGHT"

        return "CENTER"

    @staticmethod
    def _draw(frame, l_iris, r_iris, direction, avg_h, avg_v):
        cv2.circle(frame, l_iris, 5, (0, 255, 180), -1)
        cv2.circle(frame, r_iris, 5, (0, 255, 180), -1)
        cv2.rectangle(frame, (0, 0), (350, 80), (20, 20, 20), -1)
        color = (0, 220, 0) if direction != "NO FACE" else (120, 120, 120)
        cv2.putText(
            frame,
            f"Gaze: {direction}",
            (10, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"H: {avg_h:.2f}   V: {avg_v:.2f}",
            (10, 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (170, 170, 170),
            1,
            cv2.LINE_AA,
        )

    def process_frame(self, frame: np.ndarray, center_h: float = None, center_v: float = None) -> dict:
        h, w = frame.shape[:2]
        annotated = frame.copy()
        self._ts_ms += 33

        rgb_frame = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        result = self._landmarker.detect_for_video(rgb_frame, self._ts_ms)

        if not result.face_landmarks:
            return {
                "detected": False,
                "direction": "NO FACE",
                "h_ratio": 0.5,
                "v_ratio": 0.5,
                "gaze_bearing": 0.0,
                "gaze_strength": 0.0,
                "left_iris_px": None,
                "right_iris_px": None,
                "left_eye_info": None,
                "right_eye_info": None,
                "left_eye": None,
                "right_eye": None,
                "face_bbox": None,
                "annotated": annotated,
            }

        lm = result.face_landmarks[0]

        l_iris = self._iris_center_px(lm, LEFT_IRIS_CENTER, LEFT_IRIS_RING, w, h)
        l_outer = self._lm_px(lm[LEFT_EYE_OUTER], w, h)
        l_inner = self._lm_px(lm[LEFT_EYE_INNER], w, h)
        l_top = self._lm_px(lm[LEFT_EYE_TOP], w, h)
        l_bot = self._lm_px(lm[LEFT_EYE_BOTTOM], w, h)
        l_h, l_v = self._ratios(l_iris[0], l_iris[1], l_outer, l_inner, l_top, l_bot)
        l_depth = self._avg_depth(lm, LEFT_EYE_OUTER, LEFT_EYE_INNER)
        l_bearing, l_strength, l_dx, l_dy = self._gaze_vector(l_iris[0], l_iris[1], l_outer, l_inner, l_top, l_bot, "left")

        r_iris = self._iris_center_px(lm, RIGHT_IRIS_CENTER, RIGHT_IRIS_RING, w, h)
        r_outer = self._lm_px(lm[RIGHT_EYE_OUTER], w, h)
        r_inner = self._lm_px(lm[RIGHT_EYE_INNER], w, h)
        r_top = self._lm_px(lm[RIGHT_EYE_TOP], w, h)
        r_bot = self._lm_px(lm[RIGHT_EYE_BOTTOM], w, h)
        face_bbox = self._face_bbox(lm, w, h)
        yaw_off, pitch_off = self._estimate_head_pose_offsets(lm, w, h, face_bbox)

        r_h, r_v = self._ratios(r_iris[0], r_iris[1], r_outer, r_inner, r_top, r_bot)
        r_depth = self._avg_depth(lm, RIGHT_EYE_OUTER, RIGHT_EYE_INNER)
        r_bearing, r_strength, r_dx, r_dy = self._gaze_vector(r_iris[0], r_iris[1], r_outer, r_inner, r_top, r_bot, "right")
        depth_delta = l_depth - r_depth

        l_eye_w = abs(l_inner[0] - l_outer[0])
        r_eye_w = abs(r_inner[0] - r_outer[0])
        l_eye_h = abs(l_bot[1] - l_top[1])
        r_eye_h = abs(r_bot[1] - r_top[1])
        l_quality = float(l_eye_w * l_eye_h)
        r_quality = float(r_eye_w * r_eye_h)

        if abs(depth_delta) > DEPTH_PROFILE_THRESHOLD:
            dominant_is_left = l_quality >= r_quality
            avg_h = l_h if dominant_is_left else r_h
            avg_v = l_v if dominant_is_left else r_v
            gaze_bearing = l_bearing if dominant_is_left else r_bearing
            gaze_strength = l_strength if dominant_is_left else r_strength
            gaze_dx = l_dx if dominant_is_left else r_dx
            gaze_dy = l_dy if dominant_is_left else r_dy
        else:
            h_weight = l_quality / max(1e-6, l_quality + r_quality)
            avg_h = (l_h * h_weight) + (r_h * (1.0 - h_weight))
            avg_v = (l_v * h_weight) + (r_v * (1.0 - h_weight))
            gaze_bearing = (l_bearing + r_bearing) / 2.0
            gaze_strength = (l_strength + r_strength) / 2.0
            gaze_dx = (l_dx + r_dx) / 2.0
            gaze_dy = (l_dy + r_dy) / 2.0

        # During blinks/partial closure, keep prior direction to avoid random jumps.
        if min(l_eye_h, r_eye_h) < MIN_EYE_HEIGHT_PX or min(l_eye_w, r_eye_w) < MIN_EYE_WIDTH_PX:
            avg_h = self._h_history[-1] if self._h_history else 0.5
            avg_v = self._v_history[-1] if self._v_history else 0.5
            direction = self._last_direction
        else:
            # If one eye becomes noisy or the head is significantly turned, prefer the more reliable eye.
            if abs(l_h - r_h) > MAX_EYE_RATIO_DELTA:
                avg_h = l_h if l_quality >= r_quality else r_h

            if abs(l_v - r_v) > MAX_EYE_RATIO_DELTA:
                avg_v = l_v if l_quality >= r_quality else r_v

            self._h_history.append(avg_h)
            self._v_history.append(avg_v)

            # Median smoothing is robust against short outlier spikes.
            smooth_h = float(np.median(self._h_history))
            smooth_v = float(np.median(self._v_history))

            # Compensate gaze ratios for head pose drift (small correction only).
            smooth_h -= yaw_off * HEAD_POSE_YAW_GAIN
            smooth_v -= pitch_off * HEAD_POSE_PITCH_GAIN
            smooth_h = max(0.0, min(1.0, smooth_h))
            smooth_v = max(0.0, min(1.0, smooth_v))

            direction = self._classify(
                smooth_h,
                smooth_v,
                center_h=center_h,
                center_v=center_v,
                gaze_strength=gaze_strength,
                gaze_dy=gaze_dy,
            )
            avg_h, avg_v = smooth_h, smooth_v

        if center_h is None and center_v is None and not self._calib_done:
            # Only learn a baseline from stable, center-like frames.
            if gaze_strength < 0.12 and abs(avg_h - 0.5) < 0.10 and abs(avg_v - 0.5) < 0.10:
                self._calib_samples.append((avg_h, avg_v))
            if len(self._calib_samples) >= self._calib_needed:
                hs = [s[0] for s in self._calib_samples]
                vs = [s[1] for s in self._calib_samples]
                self._center_h = float(np.mean(hs))
                self._center_v = float(np.mean(vs))
                self._calib_done = True

        self._last_direction = direction
        self._draw(annotated, l_iris, r_iris, direction, avg_h, avg_v)

        left_eye_info = {
            "iris_center": list(l_iris),
            "h_ratio": round(l_h, 4),
            "v_ratio": round(l_v, 4),
            "outer": list(l_outer),
            "inner": list(l_inner),
            "top": list(l_top),
            "bottom": list(l_bot),
        }
        right_eye_info = {
            "iris_center": list(r_iris),
            "h_ratio": round(r_h, 4),
            "v_ratio": round(r_v, 4),
            "outer": list(r_outer),
            "inner": list(r_inner),
            "top": list(r_top),
            "bottom": list(r_bot),
        }

        return {
            "detected": True,
            "direction": direction,
            "h_ratio": round(avg_h, 4),
            "v_ratio": round(avg_v, 4),
            "gaze_bearing": round(gaze_bearing, 4),
            "gaze_strength": round(gaze_strength, 4),
            "gaze_dx": round(gaze_dx, 4),
            "gaze_dy": round(gaze_dy, 4),
            "eye_depth_delta": round(depth_delta, 4),
            "dominant_eye": "left" if (abs(depth_delta) > DEPTH_PROFILE_THRESHOLD and l_quality >= r_quality) else ("right" if abs(depth_delta) > DEPTH_PROFILE_THRESHOLD else "both"),
            "left_iris_px": list(l_iris),
            "right_iris_px": list(r_iris),
            "left_eye_info": left_eye_info,
            "right_eye_info": right_eye_info,
            "left_eye": {
                "iris_center": l_iris,
                "h_ratio": l_h,
                "v_ratio": l_v,
            },
            "right_eye": {
                "iris_center": r_iris,
                "h_ratio": r_h,
                "v_ratio": r_v,
            },
            "face_bbox": face_bbox,
            "annotated": annotated,
        }

    def release(self):
        self._landmarker.close()

    def recalibrate(self):
        self._calib_samples = []
        self._calib_done = False
        self._center_h = 0.5
        self._center_v = 0.5
