import torch
import torch.onnx
import onnx
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelLoader:
    """
    Model loader class for loading and saving PyTorch models.

    Args:
        model (torch.nn.Module): The PyTorch model to load/save.
        save_path (str): Path to save/load the model weights.
    """

    def __init__(self, model: torch.nn.Module = None, device: str = 'cpu'):
        self.model = model
        self.device = device

    def save(self, save_path: str):
        """Save the model weights to disk."""
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create directory if needed
            torch.save(self.model.state_dict(), save_path)
            logger.info(f"Saved model weights → {save_path}")
            return save_path
        except Exception as e:
            logger.error(f"Failed to save model weights: {e}")
            raise

    def load(self, save_path: str):
        """Load the model weights from disk."""
        try:
            if not os.path.exists(save_path):
                raise FileNotFoundError(f"Model file not found at {save_path}")
            self.model.to(self.device) 
            self.model.load_state_dict(torch.load(save_path, map_location=self.device))
            self.model.eval()  
            logger.info(f"Loaded model weights → {save_path}")
            return self.model
            
        except Exception as e:
            logger.error(f"Failed to load model weights: {e}")
            raise

    def load_weights(self, path: str):
        """Load model weights from a specified path."""
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found at {path}")
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            logger.info(f"Loaded model weights → {path}")
            return self.model
        except Exception as e:
            logger.error(f"Failed to load model weights: {e}")
            raise

    def load_onnx(self, path: str):
        """Load model weights from an ONNX file."""
        try:
            import onnxruntime as ort
            if not os.path.exists(path):
                raise FileNotFoundError(f"ONNX model file not found at {path}")
            session = ort.InferenceSession(path)
            logger.info(f"Loaded ONNX model → {path}")
            return session
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise

    def load_torchscript(self, path: str):
        """Load model weights from a TorchScript file."""
        try:
            self.model = torch.jit.load(path)
            logger.info(f"Loaded TorchScript model → {path}")
            return self.model
        except Exception as e:
            logger.error(f"Failed to load TorchScript model: {e}")
            raise

    def save_weights(self, path: str):
        """Save model weights to a specified path."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(self.model.state_dict(), path)
            logger.info(f"Saved model weights → {path}")
            return path
        except Exception as e:
            logger.error(f"Failed to save model weights: {e}")
            raise

class ModelConversion:
    """
    Model converter class for converting PyTorch models to ONNX or TorchScript formats.

    Args:
        model (torch.nn.Module): The PyTorch model to convert.
        save_path (str): Path to save the converted model.
    """

    def __init__(self, model: torch.nn.Module, save_path: str):
        self.model = model
        self.save_path = save_path
        self.model.eval()  # Set model to evaluation mode for consistent exports

    def convert_to_onnx(self, input_shape=(1, 3, 224, 224), opset_version=11):
        """
        Convert the model to ONNX format.

        Args:
            input_shape (tuple): Shape of the dummy input (e.g., (batch_size, channels, height, width)).
            opset_version (int): ONNX opset version for compatibility.
        """
        try:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            dummy_input = torch.randn(*input_shape).to('cpu')  # Ensure input is on CPU
            self.model.to('cpu')  # Ensure model is on CPU
            torch.onnx.export(
                self.model,
                dummy_input,
                self.save_path,
                opset_version=opset_version,
                input_names=["input"],
                output_names=["embedding"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "embedding": {0: "batch_size"}
                }
            )
            # Validate ONNX model
            onnx_model = onnx.load(self.save_path)
            onnx.checker.check_model(onnx_model)
            logger.info(f"Converted and validated model to ONNX format → {self.save_path}")
            return self.save_path
        except Exception as e:
            logger.error(f"Failed to convert to ONNX: {e}")
            raise

    def convert_to_torchscript(self, input_shape=(1, 3, 224, 224), method="trace"):
        """
        Convert the model to TorchScript format.

        Args:
            input_shape (tuple): Shape of the dummy input.
            method (str): Conversion method ('trace' or 'script').
        """
        try:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            dummy_input = torch.randn(*input_shape).to('cpu')
            self.model.to('cpu')
            if method == "trace":
                traced_model = torch.jit.trace(self.model, dummy_input)
            elif method == "script":
                traced_model = torch.jit.script(self.model)
            else:
                raise ValueError("Method must be 'trace' or 'script'")
            traced_model.save(self.save_path)
            logger.info(f"Converted model to TorchScript format ({method}) → {self.save_path}")
            return self.save_path
        except Exception as e:
            logger.error(f"Failed to convert to TorchScript: {e}")
            raise

if __name__ == "__main__":
    from models.embedding_net import EmbeddingNet  # Replace with your actual model import
    import warnings
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)  # Suppress tracing warnings

    model = EmbeddingNet()
    
    # model_loader = ModelLoader(model, "app/checkpoints/model_best_vggface_03.pth")
    # model_loader.load()  # Load weights from .pth file
    
    # converter = ModelConvert(model, "app/checkpoints/model_best_vggface_03.onnx")
    # converter.convert_to_onnx(input_shape=(1, 3, 224, 224), opset_version=11)
    
    # converter = ModelConvert(model, "app/checkpoints/model_best_vggface_03.pt")
    # converter.convert_to_torchscript(input_shape=(1, 3, 224, 224), method="trace")

    model_onnx = ModelLoader(model, "app/checkpoints/model_best_vggface_03.onnx")
    model_onnx.load_onnx("app/checkpoints/model_best_vggface_03.onnx")

    print("Model loaded successfully.")

