from .image_processing import ImageProcessor
from io import BytesIO
import logging

# ─── HELPERS ───────────────────────────────────────────────────────────────────
def detect_faces(image, detector):
    return detector.detect_and_crop(image)

def detect_objects(image, detector):
    results = detector.predict(image)
    boxes = results["boxes"]
    confidences = results["confidences"]
    class_ids = results["class_ids"]
    names = detector._object_name()

    detected = []
    for i, box in enumerate(boxes):
        try:
            class_idx = int(class_ids[i])
            label = names.get(class_idx, "Unknown")
            confidence = float(confidences[i])
            detected.append({
                "label": label,
                "box": box.tolist(),
                "conf": confidence,
                "class_id": class_idx
            })
        except Exception as e:
            print(f"Error processing box {i}: {e}")
    return detected

def process_image_and_get_embedding(upload_file, detector, embed_model, device):
    """Process an uploaded image, detect face, and return embedding and status."""
    if hasattr(upload_file, "file"):
        file_obj = upload_file.file
    else:
        file_obj = upload_file

    if hasattr(file_obj, "read"):
        file_bytes = file_obj.read()
        file_obj.seek(0)
    else:
        file_bytes = file_obj

    img = ImageProcessor.open_image(BytesIO(file_bytes))
    img = ImageProcessor.convert_to_npy(img)
    face_result = detect_faces(img, detector)
    
    embedding = None
    face_boxes = []

    if face_result["status"] == "multi_face":
        logging.warning("Multiple faces detected.")
        face_boxes = face_result["boxes"].tolist() if hasattr(face_result["boxes"], "tolist") else face_result["boxes"]

    elif face_result["status"] == "ok":
        logging.info("Face detected.")
        face_boxes = face_result["boxes"].tolist() if hasattr(face_result["boxes"], "tolist") else face_result["boxes"]
        embedding = ImageProcessor.get_face_embedding(face_result["face"], embed_model, device)

    return {
        "face_count": face_result.get("count", 0),
        "conf": face_result.get("confidence", 0),
        "embedding": embedding.tolist() if embedding is not None else None,
        "face_boxes": face_boxes,
        "status": face_result.get("status", "no_face")
    }