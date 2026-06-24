import torch
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import logging
from typing import Union, List, Tuple

# Configure logging for MLOps monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelInference:
    """
    Model inference class for generating embeddings using PyTorch or ONNX models.

    Args:
        model (torch.nn.Module, optional): PyTorch model for inference.
        onnx_path (str, optional): Path to the ONNX model file.
        device (str): Device for PyTorch inference ('cpu' or 'cuda').
    """

    def __init__(self, model: torch.nn.Module = None, onnx_path: str = None, device: str = 'cpu'):
        if model is None and onnx_path is None:
            raise ValueError("Either a PyTorch model or ONNX model path must be provided.")
        
        self.model = model
        self.onnx_path = onnx_path
        self.device = device
        self.ort_session = None

        # Initialize PyTorch model
        if self.model is not None:
            self.model.eval()  # Set to evaluation mode
            self.model.to(self.device)
            logger.info(f"Initialized PyTorch model on {self.device}")

        # Initialize ONNX Runtime session
        if self.onnx_path is not None:
            try:
                self.ort_session = ort.InferenceSession(self.onnx_path)
                logger.info(f"Initialized ONNX model from {self.onnx_path}")
            except Exception as e:
                logger.error(f"Failed to load ONNX model: {e}")
                raise

    def preprocess_image(self, image: Union[str, bytes, Image.Image], 
                       target_size: Tuple[int, int] = (160, 160)) -> np.ndarray:
        """
        Preprocess an input image for inference.

        Args:
            image: Path to image file, bytes, or PIL Image.
            target_size: Target size for resizing (width, height).

        Returns:
            Preprocessed image as a numpy array (batch_size, channels, height, width).
        """
        try:
            if isinstance(image, str):
                img = Image.open(image).convert('RGB')
            elif isinstance(image, bytes):
                img = Image.open(io.BytesIO(image)).convert('RGB')
            elif isinstance(image, Image.Image):
                img = image.convert('RGB')
            else:
                raise ValueError("Image must be a file path, bytes, or PIL Image")

            # Resize and normalize
            img = img.resize(target_size)
            img_array = np.array(img).transpose(2, 0, 1)  # HWC to CHW
            img_array = img_array.astype(np.float32) / 255.0  # Normalize to [0, 1]
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            return img_array
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise

    def infer_pytorch(self, image: Union[str, bytes, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Perform inference using the PyTorch model.

        Args:
            image: Input image (file path, bytes, PIL Image, or preprocessed numpy array).

        Returns:
            Embedding as a numpy array.
        """
        if self.model is None:
            raise ValueError("PyTorch model not provided for inference")

        try:
            # Preprocess if input is not already a numpy array
            if not isinstance(image, np.ndarray):
                input_data = self.preprocess_image(image)
            else:
                input_data = image

            # Convert to torch tensor and move to device
            input_tensor = torch.from_numpy(input_data).to(self.device)

            # Run inference
            with torch.no_grad():
                embedding = self.model(input_tensor).cpu().numpy()
            logger.info(f"PyTorch inference completed, embedding shape: {embedding.shape}")
            return embedding
        except Exception as e:
            logger.error(f"PyTorch inference failed: {e}")
            raise

    def infer_onnx(self, image: Union[str, bytes, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Perform inference using the ONNX model.

        Args:
            image: Input image (file path, bytes, PIL Image, or preprocessed numpy array).

        Returns:
            Embedding as a numpy array.
        """
        if self.ort_session is None:
            raise ValueError("ONNX model not provided for inference")

        try:
            # Preprocess if input is not already a numpy array
            if not isinstance(image, np.ndarray):
                input_data = self.preprocess_image(image)
            else:
                input_data = image

            # Run ONNX inference
            ort_inputs = {self.ort_session.get_inputs()[0].name: input_data}
            embedding = self.ort_session.run(None, ort_inputs)[0]
            logger.info(f"ONNX inference completed, embedding shape: {embedding.shape}")
            return embedding
        except Exception as e:
            logger.error(f"ONNX inference failed: {e}")
            raise

    def infer_batch(self, images: List[Union[str, bytes, Image.Image]], 
                    use_onnx: bool = False) -> np.ndarray:
        """
        Perform batch inference on a list of images.

        Args:
            images: List of images (file paths, bytes, or PIL Images).
            use_onnx: If True, use ONNX model; otherwise, use PyTorch model.

        Returns:
            Batch of embeddings as a numpy array.
        """
        try:
            # Preprocess all images
            input_batch = np.stack([self.preprocess_image(img).squeeze(0) for img in images])
            
            if use_onnx:
                return self.infer_onnx(input_batch)
            else:
                return self.infer_pytorch(input_batch)
        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            raise

if __name__ == "__main__":
    from models.embedding_net import EmbeddingNet  # Replace with your actual model import
    from model_loader import ModelLoader  # Replace with your actual model loader import
    import warnings

    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
    
    # Initialize model and load weights
    model = EmbeddingNet()
    model_loader = ModelLoader(model, "app/checkpoints/model_best_vggface_03.pth")
    model_loader.load()

    # Initialize inference class (PyTorch and ONNX)
    inference = ModelInference(model=model, onnx_path="app/checkpoints/model_best_vggface_03.onnx", device='cpu')

    # Example: Single image inference
    image_path = "samples/image_male_03.jpg"  # Replace with actual image path
    pytorch_embedding = inference.infer_pytorch(image_path)
    onnx_embedding = inference.infer_onnx(image_path)
    print(f"PyTorch embedding shape: {pytorch_embedding.shape}")
    print(f"ONNX embedding shape: {onnx_embedding.shape}")

    # Example: Batch inference
    image_paths = ["samples/image_male_03.jpg", "samples/image_female_02.jpeg"]  # Replace with actual paths
    batch_embeddings = inference.infer_batch(image_paths, use_onnx=True)
    print(f"Batch embeddings shape: {batch_embeddings.shape}")