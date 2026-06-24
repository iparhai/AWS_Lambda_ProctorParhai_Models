import cv2
import numpy as np
from PIL import Image
import logging

class ImageProcessor:
    """Utility class for image processing tasks."""

    @staticmethod
    def pil_to_cv2(pil_img):
        """Convert PIL Image to OpenCV BGR image."""
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    @staticmethod
    def cv2_to_pil(cv_img):
        """Convert OpenCV BGR image to PIL Image."""
        return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

    @staticmethod
    def crop_face(img, box):
        """Crop face region from image using bounding box."""
        x1, y1, x2, y2 = box
        return img[y1:y2, x1:x2]

    @staticmethod
    def draw_boxes(img, boxes, labels=None, color=(0, 255, 0)):
        """Draw bounding boxes (and optional labels) on image."""
        img_draw = img.copy()
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
            if labels and i < len(labels):
                cv2.putText(img_draw, str(labels[i]), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return img_draw

    @staticmethod
    def to_tensor(face_rgb):
        """Transform face image to tensor for embedding."""
        import torchvision.transforms as T
        tf = T.Compose([
            T.ToPILImage(),
            T.Resize((160, 160)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])
        return tf(face_rgb).unsqueeze(0)
    
    @staticmethod
    def get_face_embedding(face_img, embed_model, device):
        """
        Get face embedding using either a PyTorch model or an ONNX InferenceSession.
        """
        import torch

        from torchvision import transforms
        preprocess = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        if isinstance(face_img, np.ndarray):
            from PIL import Image
            face_img = Image.fromarray(face_img)
        tensor = preprocess(face_img).unsqueeze(0)

        if hasattr(embed_model, "forward") or callable(embed_model):
            tensor = tensor.to(device)
            with torch.no_grad():
                emb = embed_model(tensor).cpu().numpy()[0]
            return emb

        # ONNX InferenceSession
        import onnxruntime as ort
        if isinstance(embed_model, ort.InferenceSession):
            input_name = embed_model.get_inputs()[0].name
            input_data = tensor.cpu().numpy().astype(np.float32)
            outputs = embed_model.run(None, {input_name: input_data})
            emb = outputs[0][0]
            return emb

        raise TypeError("embed_model must be a PyTorch model or an ONNX InferenceSession.")

    @staticmethod
    def open_image(file):
        """Open image file and convert to RGB."""
        return Image.open(file).convert("RGB")
    
    @staticmethod
    def save_image(image, path):
        """Save image to specified path."""
        cv2.imwrite(path, image)

    @staticmethod
    def convert_to_npy(image):
        """Convert image to numpy array."""
        return np.array(image)
    
class YoloDetector:
    """Utility class for YOLO object detection."""

    def __init__(self, model_path: str, is_onnx=False):
        self.model = None
        self.model = self._load_yolo_model(model_path, is_onnx)
        
        self.is_face = "face" in model_path

    @staticmethod
    def _load_yolo_model(model_path, is_onnx):
        """Load YOLO model from the specified path."""
        from ultralytics import YOLO
        return YOLO(model_path, task='detect' if is_onnx else None)

    def _object_name(self):
        if hasattr(self.model, "model") and hasattr(self.model.model, "names"):
            return self.model.model.names
        if hasattr(self.model, "names"):
            return self.model.names
        return []

    def to_onnx(self, output_path: str):
        """Convert YOLO model to ONNX format."""
        logging.info(f"Converting YOLO model to ONNX format at {output_path}")
        self.model.export(format="onnx")

    def predict(self, image):
        results = self.model(image, conf=0.5)
        result = results[0]
        return {
            "boxes": result.boxes.xyxy.cpu().numpy(),
            "confidences": result.boxes.conf.cpu().numpy(),
            "class_ids": result.boxes.cls.cpu().numpy()
        }

    def detect_and_crop(self, image):
        prediction = self.predict(image)
        boxes = prediction["boxes"]
        confidences = prediction["confidences"]
        class_ids = prediction["class_ids"]

        if len(boxes) == 0:
            return {"status": "no_face", "count": 0, "boxes": boxes, "confidences": confidences, "class_ids": class_ids}
        elif len(boxes) > 1 and self.is_face:
            return {"status": "multi_face", "count": len(boxes), "boxes": boxes, "confidences": confidences, "class_ids": class_ids}

        objects = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cropped = image[y1:y2, x1:x2]
            objects.append(cropped)

        return {
            "status": "ok",
            "count": len(objects),
            "face": objects[0] if objects else None,
            "objects": objects,
            "boxes": boxes,
            "confidences": confidences,
            "class_ids": class_ids
        }