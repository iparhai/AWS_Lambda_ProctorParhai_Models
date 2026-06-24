"""
main.py
FastAPI Lambda handler for MediaPipe-based gaze detection.
Supports both /vision and /vision/ endpoints.
"""

import json
import logging
from typing import Dict, List, Optional, Union

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mangum import Mangum
from pydantic import BaseModel

from src.gaze_tracking import IrisTracker
from utils import decode_image, build_response, build_error_response

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Gaze Detection API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── singleton tracker (warm between Lambda invocations) ──────────────────────
_tracker: Optional[IrisTracker] = None

def get_tracker() -> IrisTracker:
    global _tracker
    if _tracker is None:
        logger.info("Initialising IrisTracker …")
        _tracker = IrisTracker()
        logger.info("IrisTracker ready.")
    return _tracker


# ── request schema ───────────────────────────────────────────────────────────
class GazeRequest(BaseModel):
    image_data: str
    screen_size: Optional[Union[List[int], Dict[str, int]]] = None
    detection_types: List[str] = ["gaze_detection"]
    gaze_thres: Optional[float] = None
    detector_model: Optional[str] = "dlib"   # accepted for backward compatibility, ignored internally
    bounding_box: Optional[list] = None
    is_glasses: Optional[bool] = False
    # optional calibration baseline — pass if you've pre-calibrated client-side
    center_h: Optional[float] = None
    center_v: Optional[float] = None


def _resolve_screen_size(screen_size: Optional[Union[List[int], Dict[str, int]]], frame) -> List[int]:
    frame_h, frame_w = frame.shape[:2]

    if isinstance(screen_size, list) and len(screen_size) >= 2:
        w = int(screen_size[0])
        h = int(screen_size[1])
        if w > 0 and h > 0:
            return [w, h]

    if isinstance(screen_size, dict):
        w = screen_size.get("width") or screen_size.get("w") or screen_size.get("screenWidth")
        h = screen_size.get("height") or screen_size.get("h") or screen_size.get("screenHeight")
        if w and h:
            w = int(w)
            h = int(h)
            if w > 0 and h > 0:
                return [w, h]

    # Fallback to incoming image dimensions if frontend does not send screen size.
    return [int(frame_w), int(frame_h)]


# ── shared handler ────────────────────────────────────────────────────────────
async def _handle_gaze(request: GazeRequest) -> JSONResponse:
    try:
        frame = decode_image(request.image_data)
    except ValueError as exc:
        logger.warning("Image decode error: %s", exc)
        return JSONResponse(
            status_code=400,
            content=build_error_response(str(exc), status_code=400),
        )

    effective_screen_size = _resolve_screen_size(request.screen_size, frame)

    try:
        tracker = get_tracker()
        result  = tracker.process_frame(
            frame,
            center_h=request.center_h,
            center_v=request.center_v,
        )
    except Exception as exc:
        logger.exception("Tracker error")
        return JSONResponse(
            status_code=500,
            content=build_error_response(f"Tracker error: {exc}", status_code=500),
        )

    response_body = build_response(result)
    logger.info(
        "direction=%s h=%.3f v=%.3f screen=%sx%s",
        result["direction"],
        result.get("h_ratio", 0),
        result.get("v_ratio", 0),
        effective_screen_size[0],
        effective_screen_size[1],
    )
    return JSONResponse(status_code=200, content=response_body)


# ── routes — both /vision and /vision/ ───────────────────────────────────────
@app.post("/vision")
@app.post("/vision/")
async def vision(request: GazeRequest):
    return await _handle_gaze(request)


# Health-check
@app.get("/health")
@app.get("/health/")
async def health():
    return {"status": "ok", "model": "mediapipe"}


@app.get("/ping")
@app.get("/ping/")
async def ping():
    return {"message": "Pong!"}


# ── Lambda entrypoint ─────────────────────────────────────────────────────────
handler = Mangum(app, lifespan="off")
