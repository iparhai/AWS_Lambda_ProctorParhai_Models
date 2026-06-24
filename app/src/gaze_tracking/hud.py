import cv2
import numpy as np


_ARROWS = {
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
    "UP": (0, -1),
    "DOWN": (0, 1),
    "CENTER": (0, 0),
    "NO FACE": (0, 0),
}

_COLORS = {
    "LEFT": (50, 100, 255),
    "RIGHT": (50, 100, 255),
    "UP": (255, 140, 50),
    "DOWN": (255, 140, 50),
    "CENTER": (50, 220, 50),
    "NO FACE": (120, 120, 120),
}


def draw_direction_hud(frame: np.ndarray, direction: str, fps: float = 0.0) -> None:
    h, w = frame.shape[:2]
    panel_size = 120
    cx = w - panel_size // 2 - 10
    cy = h - panel_size // 2 - 10

    cv2.rectangle(
        frame,
        (w - panel_size - 20, h - panel_size - 20),
        (w - 2, h - 2),
        (25, 25, 25),
        -1,
    )

    for angle_deg in range(0, 360, 45):
        angle = np.deg2rad(angle_deg)
        x1 = int(cx + 40 * np.cos(angle))
        y1 = int(cy + 40 * np.sin(angle))
        x2 = int(cx + 48 * np.cos(angle))
        y2 = int(cy + 48 * np.sin(angle))
        cv2.line(frame, (x1, y1), (x2, y2), (80, 80, 80), 1)

    cv2.circle(frame, (cx, cy), 50, (60, 60, 60), 1)

    color = _COLORS.get(direction, (200, 200, 200))
    dx, dy = _ARROWS.get(direction, (0, 0))

    if direction == "CENTER":
        cv2.circle(frame, (cx, cy), 10, color, -1)
    elif direction == "NO FACE":
        cv2.putText(frame, "?", (cx - 8, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    else:
        tip = (int(cx + dx * 38), int(cy + dy * 38))
        base = (int(cx - dx * 15), int(cy - dy * 15))
        cv2.arrowedLine(frame, base, tip, color, 4, cv2.LINE_AA, tipLength=0.35)

    cv2.putText(
        frame,
        direction,
        (w - panel_size - 18, h - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )

    if fps > 0:
        cv2.putText(
            frame,
            f"FPS {fps:.1f}",
            (w - 95, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )
