"""
utils.py
Image decoding and response-building utilities.
Matches the existing gaze Lambda response schema exactly.
"""

import base64
import re
import numpy as np
import cv2


# ── direction mapping ────────────────────────────────────────────────────────
# Maps MediaPipe direction → legacy "gaze" field value + gaze_direction
_DIRECTION_MAP = {
    "CENTER": ("On Screen", "center"),
    "LEFT":   ("On Screen", "left"),
    "RIGHT":  ("On Screen", "right"),
    "UP":     ("On Screen", "up"),
    "DOWN":   ("Off Screen", "down"),
    "NO FACE": ("Gaze Not Detected", "unknown"),
}


def _normalize_direction(direction: str, detected: bool):
    if not detected:
        return "Gaze Not Detected", "unknown"

    raw = str(direction or "").upper().replace("_", " ").replace("-", " ")

    if raw in _DIRECTION_MAP:
        return _DIRECTION_MAP[raw]

    # Handle non-canonical values like "DOWN AND LEFT" from older clients.
    if "DOWN" in raw:
        return "Off Screen", "down"
    if "UP" in raw:
        return "On Screen", "up"
    if "LEFT" in raw:
        return "On Screen", "left"
    if "RIGHT" in raw:
        return "On Screen", "right"
    if "CENTER" in raw:
        return "On Screen", "center"

    return "Gaze Not Detected", "unknown"


def decode_image(image_data: str) -> np.ndarray:
    """
    Decode a base64 image string (with or without data-URI prefix) to a BGR numpy array.
    Raises ValueError on failure.
    """
    # Strip data-URI prefix if present  e.g. "data:image/jpeg;base64,/9j/..."
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    # Remove any whitespace / line-breaks that might have crept in
    image_data = re.sub(r"\s+", "", image_data)

    try:
        img_bytes = base64.b64decode(image_data)
    except Exception as exc:
        raise ValueError(f"base64 decode failed: {exc}")

    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("cv2.imdecode returned None — invalid image bytes")
    return frame


def build_response(tracker_result: dict) -> dict:
    """
    Convert IrisTracker.process_frame() output into the canonical
    Lambda gaze_response JSON schema.
    """
    detected  = tracker_result["detected"]
    direction = tracker_result["direction"]      # UP/DOWN/LEFT/RIGHT/CENTER/NO FACE

    gaze_legacy, gaze_direction = _normalize_direction(direction, detected)

    # pupil coords  (absolute pixel positions)
    l_px = tracker_result.get("left_iris_px")
    r_px = tracker_result.get("right_iris_px")

    pupil_coord = {
        "left":  l_px if l_px else [0, 0],
        "right": r_px if r_px else [0, 0],
    }

    # ratios [horizontal, vertical]
    ratios = [
        tracker_result.get("h_ratio", 0.5),
        tracker_result.get("v_ratio", 0.5),
    ]

    gaze_metrics = {
        "bearing": tracker_result.get("gaze_bearing", 0.0),
        "strength": tracker_result.get("gaze_strength", 0.0),
        "gaze_dx": tracker_result.get("gaze_dx", 0.0),
        "gaze_dy": tracker_result.get("gaze_dy", 0.0),
        "eye_depth_delta": tracker_result.get("eye_depth_delta", 0.0),
        "dominant_eye": tracker_result.get("dominant_eye", "both"),
    }

    # face bounding box
    face_bbox = tracker_result.get("face_bbox")

    # eye detail blocks (mirrors old dlib schema as closely as possible)
    left_info  = tracker_result.get("left_eye_info")
    right_info = tracker_result.get("right_eye_info")

    def _eye_block(info, side_label):
        if not info:
            return None

        center = info.get("iris_center") or [0, 0]
        outer = info.get("outer") or center
        inner = info.get("inner") or center
        top = info.get("top") or center
        bottom = info.get("bottom") or center

        x1 = int(min(outer[0], inner[0]))
        y1 = int(min(top[1], bottom[1]))
        x2 = int(max(outer[0], inner[0]))
        y2 = int(max(top[1], bottom[1]))
        width = float(max(0, x2 - x1))
        height = float(max(0, y2 - y1))

        local_x = int(center[0] - x1)
        local_y = int(center[1] - y1)

        return {
            "pupil_x": local_x,
            "pupil_y": local_y,
            "center": center,
            "h_ratio": info.get("h_ratio"),
            "v_ratio": info.get("v_ratio"),
            "bounding_box": [x1, y1, int(width), int(height)],
            "width": width,
            "height": height,
            "origin": [x1, y1],
        }

    eye_block = {
        "left_pupil":  _eye_block(left_info,  "left"),
        "right_pupil": _eye_block(right_info, "right"),
    }

    return {
        "gaze_response": {
            # ── legacy fields (backward compat) ──────────────────────────
            "gaze":           gaze_legacy,
            "bounding_box":   face_bbox,
            "ratios":         ratios,
            "status_code":    200,

            # ── new fields ───────────────────────────────────────────────
            "gaze_direction": gaze_direction,
            "pupil_coord":    pupil_coord,
            "eye":            eye_block,
            "gaze_metrics":   gaze_metrics,

            # ── face block ───────────────────────────────────────────────
            "face": {
                "is_detected":   detected,
                "bounding_box":  face_bbox,
                "error_message": None,
            },
        }
    }


def build_error_response(message: str, status_code: int = 400) -> dict:
    return {
        "gaze_response": {
            "gaze":           "Gaze Not Detected",
            "gaze_direction": "unknown",
            "pupil_coord":    {"left": [0, 0], "right": [0, 0]},
            "ratios":         [0.5, 0.5],
            "status_code":    status_code,
            "face": {
                "is_detected":   False,
                "bounding_box":  None,
                "error_message": message,
            },
            "eye": {
                "left_pupil":  None,
                "right_pupil": None,
            },
            "gaze_metrics": {
                "bearing": 0.0,
                "strength": 0.0,
                "gaze_dx": 0.0,
                "gaze_dy": 0.0,
                "eye_depth_delta": 0.0,
                "dominant_eye": "both",
            },
            "bounding_box": None,
        }
    }
