import torch
from tqdm import tqdm
import mlflow

class Trainer:
    """
    Trainer class for model training and validation with MLflow logging.

    Args:
        model (torch.nn.Module): The model to train.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        train_loader (torch.utils.data.DataLoader): Training data loader.
        val_loader (torch.utils.data.DataLoader, optional): Validation data loader.
        num_epochs (int): Number of epochs.
        save_path (str): Path to save the model.
        patience (int): Early stopping patience.
        clip_value (float): Gradient clipping value.
        mlflow_experiment (str): MLflow experiment name.
        mlflow_run_name (str, optional): MLflow run name.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader = None,
        num_epochs: int = 10,
        save_path: str = "model.pth",
        patience: int = 5,
        clip_value: float = 1.0,
        mlflow_experiment: str = "face-triplet",
        mlflow_run_name: str = None
    ):
        # core objects
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader

        # hyperparams
        self.num_epochs = num_epochs
        self.patience = patience
        self.clip_value = clip_value
        self.save_path = save_path

        # training state
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_history = []
        self.val_history = []

        # MLflow settings
        self.mlflow_experiment = mlflow_experiment
        self.mlflow_run_name = mlflow_run_name

        log_params = {
            "num_epochs": self.num_epochs,
            "patience": self.patience,
            "clip_value": self.clip_value,
            "batch_size": self.train_loader.batch_size,
            "lr": self.optimizer.param_groups[0]['lr']
        }

        # move to device
        self.model.to(self.device)
        self.criterion.to(self.device)

    def _train_one_epoch(self, epoch: int) -> tuple:
        """
        Train the model for one epoch.

        Args:
            epoch (int): Current epoch number.

        Returns:
            tuple: (average_loss, list_of_batch_losses)
        """
        self.model.train()
        batch_losses = []
        pbar = tqdm(self.train_loader, desc=f" Train Epoch {epoch+1}/{self.num_epochs}")

        batch_losses = self._each_batch(pbar, istype="train", epoch=epoch, batch_losses=batch_losses)

        avg = sum(batch_losses) / len(batch_losses)
        self.train_history.append(batch_losses)
        print(f"→ Avg Train Loss epoch {epoch+1}: {avg:.4f}")
        return avg, batch_losses

    def _each_batch(self, pbar, istype: str = "train", epoch: int = 0, batch_losses: list = []) -> list:
        """
        Process each batch for training or validation.

        Args:
            pbar: tqdm progress bar.
            istype (str): "train" or "val".
            epoch (int): Current epoch.
            batch_losses (list): List to store batch losses.

        Returns:
            list: Updated batch_losses.
        """
        for batch_idx, (a, p, n) in enumerate(pbar):
            a, p, n = a.to(self.device), p.to(self.device), n.to(self.device)
            out_a = self.model(a)
            out_p = self.model(p)
            out_n = self.model(n)
            loss = self.criterion(out_a, out_p, out_n)

            if torch.isnan(loss):
                print(f"NaN at {istype} batch {batch_idx}, skipping")
                continue

            l = loss.item()
            batch_losses.append(l)

            if istype == "train":
                pbar.set_postfix(train_loss=f"{l:.4f}")
            else:
                pbar.set_postfix(val_loss=f"{l:.4f}")

        return batch_losses

    def _validate(self, epoch: int) -> tuple:
        """
        Validate the model for one epoch.

        Args:
            epoch (int): Current epoch number.

        Returns:
            tuple: (average_loss, list_of_batch_losses)
        """
        if self.val_loader is None:
            return None, []
        self.model.eval()
        
        batch_losses = []
        pbar = tqdm(self.val_loader, desc=f" Val Epoch {epoch+1}/{self.num_epochs}")

        with torch.no_grad():
            batch_losses = self._each_batch(pbar, istype="val", epoch=epoch, batch_losses=batch_losses)

        avg = sum(batch_losses) / len(batch_losses)
        self.val_history.append(batch_losses)
        
        print(f"→ Avg Val Loss epoch {epoch+1}: {avg:.4f}")
        return avg, batch_losses

    def save_best_model(self) -> str:
        """
        Save the best model to disk.

        Returns:
            str: Path to the saved model.
        """
        best_path = self.save_path.replace(".pth", "_best.pth")
        torch.save(self.model.state_dict(), best_path)
        print(f"[+] Saved best model → {best_path}")
        return best_path

    def train(self) -> torch.nn.Module:
        """
        Train the model with early stopping and MLflow logging.

        Returns:
            torch.nn.Module: The best model after training.
        """
        mlflow.set_experiment(self.mlflow_experiment)
        with mlflow.start_run(run_name=self.mlflow_run_name):
            mlflow.log_params(self.log_params)
            mlflow.start_run(run_name=self.mlflow_run_name)
            
            for epoch in range(self.num_epochs):
                avg_train, train_batch_losses = self._train_one_epoch(epoch)
                for batch_idx, l in enumerate(train_batch_losses):
                    mlflow.log_metric("train_batch_loss", l, step=epoch * len(self.train_loader) + batch_idx)

                mlflow.log_metric("train_epoch_loss", avg_train, step=epoch)
                
                avg_val, val_batch_losses = self._validate(epoch)
                if val_batch_losses:
                    for batch_idx, l in enumerate(val_batch_losses):
                        mlflow.log_metric("val_batch_loss", l, step=epoch * len(self.val_loader) + batch_idx)
                    
                    mlflow.log_metric("val_epoch_loss", avg_val, step=epoch)
                
                # early stopping logic
                if avg_val is not None and avg_val < self.best_val_loss:
                    self.best_val_loss = avg_val
                    self.epochs_no_improve = 0
                    best_path = self.save_best_model()
                    mlflow.log_artifact(best_path)
                else:
                    self.epochs_no_improve += 1
                    print(f"[!] No improvement for {self.epochs_no_improve} epochs")
                
                
                if self.epochs_no_improve >= self.patience:
                    print("[!] Early stopping")
                    break

            best_path = self.save_path.replace(".pth", "_best.pth")
            self.model.load_state_dict(torch.load(best_path, map_location=self.device))
            print("[*] Training complete, best model loaded")
            return self.model
    
    def test(self, test_loader: torch.utils.data.DataLoader) -> tuple:
        """
        Test the model on a test dataset.

        Args:
            test_loader (torch.utils.data.DataLoader): Test data loader.

        Returns:
            tuple: (average_loss, list_of_batch_losses)
        """
        self.model.eval()
        batch_losses = []
        pbar = tqdm(test_loader, desc="Test")

        with torch.no_grad():
            batch_losses = self._each_batch(pbar, istype="test", epoch=0, batch_losses=batch_losses)

        avg = sum(batch_losses) / len(batch_losses)
        print(f"→ Avg Test Loss: {avg:.4f}")
        return avg, batch_losses
    
class ModelLoader:
    """
    Model loader class for loading and saving models.

    Args:
        model (torch.nn.Module): The model to load/save.
        save_path (str): Path to save the model.
    """
    def __init__(self, model: torch.nn.Module, save_path: str):
        self.model = model
        self.save_path = save_path

    def save(self):
        """Save the model to disk."""
        torch.save(self.model.state_dict(), self.save_path)
        print(f"[+] Saved model → {self.save_path}")
        return self.save_path
    
    def load(self):
        """Load the model from disk."""
        self.model.load_state_dict(torch.load(self.save_path))
        print(f"[+] Loaded model → {self.save_path}")
        return self.model
    
    def load_weights(self, path: str):
        """Load model weights from a specified path."""
        self.model.load_state_dict(torch.load(path))
        print(f"[+] Loaded model weights → {path}")
        return self.model
    
    def save_weights(self, path: str):
        """Save model weights to a specified path."""
        torch.save(self.model.state_dict(), path)
        print(f"[+] Saved model weights → {path}")
        return path