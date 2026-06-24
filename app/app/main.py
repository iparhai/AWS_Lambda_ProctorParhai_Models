from typing import List

# Import necessary libraries
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
import numpy as np
import torch
import os

# Import custom modules
from .models.embedding_net import EmbeddingNet
from .image_processing import ImageProcessor, YoloDetector
from .model_loader import ModelLoader
from .utils import detect_objects, process_image_and_get_embedding


# ─── MODEL LOADING ─────────────────────────────────────────────────────────────
base_dir = os.path.dirname(os.path.abspath(__file__))
YOLO_FACE_WEIGHTS = os.path.join(base_dir, "checkpoints/yolov11n-face.pt")
YOLO_FACE_WEIGHTS_ONNX = os.path.join(base_dir, "checkpoints/yolov11n-face.onnx")

YOLO_OBJ_WEIGHTS = os.path.join(base_dir, "checkpoints/yolov8n-obj.pt")
YOLO_OBJ_WEIGHTS_ONNX = os.path.join(base_dir, "checkpoints/yolov8n-obj.onnx")
EMBED_WEIGHTS = os.path.join(base_dir, "checkpoints/embedding_model.onnx")

if not os.path.exists(YOLO_FACE_WEIGHTS_ONNX):
    raise FileNotFoundError(f"YOLO face weights ONNX file not found: {YOLO_FACE_WEIGHTS_ONNX}")
if not os.path.exists(YOLO_OBJ_WEIGHTS_ONNX):
    raise FileNotFoundError(f"YOLO object weights ONNX file not found: {YOLO_OBJ_WEIGHTS_ONNX}")
if not os.path.exists(EMBED_WEIGHTS):
    raise FileNotFoundError(f"Embedding weights file not found: {EMBED_WEIGHTS}")

yolo_face_detector = YoloDetector(YOLO_FACE_WEIGHTS_ONNX, is_onnx=True)
yolo_object_detector = YoloDetector(YOLO_OBJ_WEIGHTS_ONNX, is_onnx=True)

# Load embedding model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed_model = ModelLoader().load_onnx(EMBED_WEIGHTS)

# --- PyDantic model for image processing ---
from pydantic import BaseModel
from typing import List, Optional
import base64
from io import BytesIO
from PIL import Image
import numpy as np

class ImageData(BaseModel):
    filename: str = None
    content_type: str = None
    data: str


class AnalyzeRequest(BaseModel):
    input_image: str  # Base64 encoded image data
    reference_images: List[str]  # List of Base64 encoded reference images
    object_detection: Optional[bool] = False

    def validate(self):
        if not self.input_image:
            raise ValueError("Input image is required.")
        if not self.reference_images:
            raise ValueError("At least one reference image is required.")
        for ref in self.reference_images:
            if not ref:
                raise ValueError("Reference image is missing data.")
        if not isinstance(self.object_detection, bool):
            raise ValueError("Object detection flag must be a boolean value.")

from pydantic import BaseModel, Field

class AnalyzeResponse(BaseModel):
    face_count: int = 0  # Number of faces detected
    face_boxes: List[List[int]]  # List of bounding boxes for detected faces
    objects_detected: Optional[List[str]] = None  # Detected object labels
    object_boxes: Optional[List[dict]] = None  # Object detection results
    similarity: Optional[List[dict]] = None  # Similarity scores with reference images

class EmbeddingResponse(BaseModel):
    embedding: Optional[List[float]] = None
    status: str = "ok"

class EmbeddingRequest(BaseModel):
    input_image: ImageData
    use_onnx: Optional[bool] = False  # Flag to use ONNX model if available

def decode_base64_image(base64_str: str) -> Image.Image:
    image_bytes = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_bytes))


# ─── FASTAPI APP ───────────────────────────────────────────────────────────────
router = APIRouter()

@router.post("/analyze")
async def analyze(request: AnalyzeRequest):
    """ Analyze an input image for faces and objects, and compare with reference images.
    Args:
        request (AnalyzeRequest): The request containing input image, reference images, and object detection flag.
    Returns:
        dict: A response containing face count, face boxes, object detection results, and similarity scores.
    """
    result = {"face_count": 0, "face_boxes": [], "similarity": None}

    input_img_pil = decode_base64_image(request.input_image)
    input_img_np = ImageProcessor.convert_to_npy(input_img_pil)

    if request.object_detection:
        object_result = detect_objects(input_img_np, yolo_object_detector)
        result["objects_detected"] = [obj.get("label") for obj in object_result]
        result["object_boxes"] = object_result

    input_proc = process_image_and_get_embedding(BytesIO(base64.b64decode(request.input_image)), yolo_face_detector, embed_model, device)
    result.update({
        "face_count": input_proc.get("face_count"),
        "face_boxes": input_proc.get("face_boxes")
    })
    face_embedding = input_proc.get("embedding")

    ref_embs, ref_names = [], []
    for idx, ref in enumerate(request.reference_images):
        ref_proc = process_image_and_get_embedding(BytesIO(base64.b64decode(ref)), yolo_face_detector, embed_model, device)
        ref_embs.append(ref_proc.get("embedding"))
        ref_names.append(f"ref_{idx}" if ref_proc.get("status") == "ok" else f"ref_{idx} (no face)")

    if face_embedding and any(e is not None for e in ref_embs):
        result["similarity"] = [
            {
                "reference": n,
                "score": float(cosine_similarity(
                    np.array(face_embedding).reshape(1, -1),
                    np.array(ref_emb).reshape(1, -1)
                )[0, 0]) if ref_emb is not None else None
            }
            for n, ref_emb in zip(ref_names, ref_embs)
        ]

    return result


@router.post("/embedding")
async def get_embedding(request: EmbeddingRequest) -> EmbeddingResponse:
    """Get embedding for an input image.
    Args:
        request (EmbeddingRequest): The request containing input image and optional ONNX flag.  
    Returns:
        JSONResponse: A response containing the embedding and status.
    """
    input_image = decode_base64_image(request.input_image.data)

    input_proc = process_image_and_get_embedding(input_image, yolo_face_detector, embed_model, device)
    return EmbeddingResponse(
        embedding=input_proc.get("embedding"),
        status=input_proc.get("status", "ok")
    )

@router.post("/convert_to_base64", response_class=JSONResponse)
async def convert_to_base64(file: UploadFile = File(...)) -> JSONResponse:
    """
    Convert an uploaded file to a base64 string.
    
    Args:
        file (UploadFile): The file to convert.
    
    Returns:
        JSONResponse: A response containing the base64 string of the file.
    """
    try:
        contents = await file.read()
        base64_str = base64.b64encode(contents).decode('utf-8')
        return JSONResponse(content={"filename": file.filename, "data": base64_str, "content_type": file.content_type})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
