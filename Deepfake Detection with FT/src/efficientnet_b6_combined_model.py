# %%
import os
import random
import sys
import time
from datetime import datetime
import numpy as np
import multiprocessing

# Import PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader

# Import torchvision
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.models import efficientnet_b6, EfficientNet_B6_Weights
from torchvision.transforms import ToTensor, Resize, Normalize

# Import matplotlib for visualization
import matplotlib.pyplot as plt


from PIL import Image

from tqdm.auto import tqdm

# Metrics
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
from typing import Dict, Optional
import logging
from pathlib import Path


class FrequencyBranch(nn.Module):
    def __init__(self, HEIGHT, WIDTH, output_size=128, hidden_size1=512, hidden_size2=256):
        super(FrequencyBranch, self).__init__()
        input_size = 3 * HEIGHT * WIDTH * 2
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()

    def forward(self, img):
        # Perform FFT on GPU
        f_transform = torch.fft.fft2(img)
        f_transform_shifted = torch.fft.fftshift(f_transform)
        
        # Calculate amplitude and phase on GPU
        amplitude = torch.abs(f_transform_shifted)
        phase = torch.angle(f_transform_shifted)
        
        # Flatten and concatenate
        features = torch.cat((amplitude.flatten(1), phase.flatten(1)), dim=1)
        
        # Pass through network
        x = self.relu(self.fc1(features))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
    
class PreTrainedBranch(nn.Module):
    def __init__(self, input_channels=3, output_features=128):
        super(PreTrainedBranch, self).__init__()
        
        # Load pretrained EfficientNet
        self.efficientnet = efficientnet_b6(weights=EfficientNet_B6_Weights.IMAGENET1K_V1)

        # Unfreeze more layers
        for param in self.efficientnet.parameters():
            param.requires_grad = True
        
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(self.efficientnet.classifier[1].in_features, output_features),
        )
        
    def forward(self, x):
        return self.efficientnet(x)

class CombinedModel(nn.Module):
    def __init__(self, HEIGHT, WIDTH):
        super(CombinedModel, self).__init__()
        self.freq_branch = FrequencyBranch(HEIGHT, WIDTH, output_size=128)
        self.conv_branch = PreTrainedBranch(output_features=128)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Same input for both branches
        freq_output = self.freq_branch(x)
        conv_output = self.conv_branch(x)

        combined = torch.cat((freq_output, conv_output), dim=1)
        x = torch.relu(self.fc1(combined))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

class MetricsHandler:
    """
    Handles calculation and logging of various classification metrics using TorchMetrics
    and TensorBoard visualization.
    """
    def __init__(
        self,
        num_classes: int = 2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        log_dir: str = "runs/experiment"
    ):
        self.num_classes = num_classes
        self.device = device
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir)
        
        # Initialize metrics
        self._init_metrics()
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def _init_metrics(self):
        """Initialize all metrics using TorchMetrics."""
        common_kwargs = {
            "num_classes": self.num_classes,
            "task": "multiclass" if self.num_classes > 2 else "binary"
        }
        
        self.metrics = {
            'train': {
                'accuracy': torchmetrics.Accuracy(**common_kwargs).to(self.device),
                'precision': torchmetrics.Precision(**common_kwargs).to(self.device),
                'recall': torchmetrics.Recall(**common_kwargs).to(self.device),
                'f1': torchmetrics.F1Score(**common_kwargs).to(self.device),
                'auroc': torchmetrics.AUROC(**common_kwargs).to(self.device),
            },
            'val': {
                'accuracy': torchmetrics.Accuracy(**common_kwargs).to(self.device),
                'precision': torchmetrics.Precision(**common_kwargs).to(self.device),
                'recall': torchmetrics.Recall(**common_kwargs).to(self.device),
                'f1': torchmetrics.F1Score(**common_kwargs).to(self.device),
                'auroc': torchmetrics.AUROC(**common_kwargs).to(self.device),
            },
            'test': {
                'accuracy': torchmetrics.Accuracy(**common_kwargs).to(self.device),
                'precision': torchmetrics.Precision(**common_kwargs).to(self.device),
                'recall': torchmetrics.Recall(**common_kwargs).to(self.device),
                'f1': torchmetrics.F1Score(**common_kwargs).to(self.device),
                'auroc': torchmetrics.AUROC(**common_kwargs).to(self.device),
            }
        }
        
        # Add EER (Equal Error Rate) metric
        #if self.num_classes == 2:
            #for phase in ['train', 'val', 'test']:
                #self.metrics[phase]['eer'] = torchmetrics.functional.classification.binary_auroc(preds=y_preds, target=y).to(self.device)
        
        # Initialize loss tracking
        self.running_loss = {
            'train': [],
            'val': [],
            'test': []
        }

    def reset(self, phase: str):
        """Reset metrics for the given phase."""
        for metric in self.metrics[phase].values():
            metric.reset()
        self.running_loss[phase] = []

    def update(
        self,
        phase: str,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        loss: Optional[float] = None
    ):
        """
        Update metrics for the current batch.
        
        Args:
            phase: Current phase ('train', 'val', or 'test')
            outputs: Model output logits (N, num_classes)
            targets: Ground truth labels (N,)
            loss: Optional batch loss value
        """
        # Apply softmax to get probabilities
        # probs = torch.round(torch.sigmoid(outputs))
        probs = torch.sigmoid(outputs)

        
        # Update all metrics
        for metric in self.metrics[phase].values():
            # Only round for specific metrics that need binary predictions
            if isinstance(metric, (torchmetrics.Accuracy, torchmetrics.Precision, 
                             torchmetrics.Recall, torchmetrics.F1Score)):
                binary_preds = (probs > 0.5).float()  # Use 0.5 threshold
                metric.update(binary_preds, targets)
            else:
                # Use raw probabilities for other metrics
                metric.update(probs, targets)
            
        # Track loss
        if loss is not None:
            self.running_loss[phase].append(loss)

    def compute_epoch_metrics(self, phase: str, epoch: int) -> Dict[str, float]:
        """
        Compute and log all metrics for the epoch.
        
        Args:
            phase: Current phase ('train', 'val', or 'test')
            epoch: Current epoch number
            
        Returns:
            Dictionary containing all computed metrics
        """
        metrics = {}
        
        # Compute all metrics
        for metric_name, metric in self.metrics[phase].items():
            try:
                value = metric.compute()
                metrics[metric_name] = value.item()
                
                # Log to TensorBoard
                self.writer.add_scalar(
                    f'{phase}/{metric_name}',
                    metrics[metric_name],
                    epoch
                )
            except Exception as e:
                self.logger.warning(f"Failed to compute {metric_name}: {str(e)}")
        
        # Compute and log average loss
        if self.running_loss[phase]:
            avg_loss = sum(self.running_loss[phase]) / len(self.running_loss[phase])
            metrics['loss'] = avg_loss
            self.writer.add_scalar(f'{phase}/loss', avg_loss, epoch)
        
        return metrics

    def log_epoch_metrics(self, epoch: int, phase: str) -> Dict[str, float]:
        """Log metrics for current epoch."""
        metrics = self.compute_epoch_metrics(phase, epoch)
        
        metrics_str = ' | '.join([
            f'{metric}: {value:.4f}'
            for metric, value in metrics.items()
        ])
        self.logger.info(f'{phase.capitalize()} Epoch {epoch}: {metrics_str}')
        
        return metrics
    
    def close(self):
        """Close TensorBoard writer."""
        self.writer.close()


# %%
def compute_mean_and_std(data_dir):
    """
    Compute per-channel mean and std of the dataset (to be used in transforms.Normalize())
    """
    cache_file = f"./logs/mean_and_std_{os.path.basename(data_dir)}.pt"
    cache_dir = os.path.dirname(cache_file)
    
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    if os.path.exists(cache_file):
        print(f"Reusing cached mean and std")
        d = torch.load(cache_file)
        return d["mean"], d["std"]

    ds = datasets.ImageFolder(
        data_dir, transform=transforms.Compose([transforms.ToTensor()])
    )
    dl = torch.utils.data.DataLoader(
        ds, batch_size=1, num_workers=multiprocessing.cpu_count()
    )

    mean = 0.0
    for images, _ in tqdm(dl, total=len(ds), desc="Computing mean", ncols=80):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(dl.dataset)

    var = 0.0
    npix = 0
    for images, _ in tqdm(dl, total=len(ds), desc="Computing std", ncols=80):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])
        npix += images.nelement()

    std = torch.sqrt(var / (npix / 3))

    # Cache results so we don't need to redo the computation
    torch.save({"mean": mean, "std": std}, cache_file)

    return mean, std

# %%
def get_data_loaders(data_dir, batch_size, transform=None, shuffle=False, balanced=True, max_real_samples=20000, max_fake_samples=30000):
    """
    Create train and test data loaders using torchvision.datasets.ImageFolder.
    """
    
    if balanced:
        real_datset = datasets.ImageFolder(f"{data_dir}/real", transform=transform)
        fake_dataset = datasets.ImageFolder(f"{data_dir}/fake", transform=transform)

        real_indices = torch.randperm(len(real_datset))[:max_real_samples]
        fake_indices = torch.randperm(len(fake_dataset))[:max_fake_samples]

        dataset = torch.utils.data.Subset(real_datset, real_indices) + torch.utils.data.Subset(fake_dataset, fake_indices)

    else:
        BATCHES_PER_EPOCH = 200

        dataset = datasets.ImageFolder(data_dir, transform=transform)
        if data_dir.rstrip('/').endswith('train'):
            train_indices = torch.randperm(len(dataset))[:BATCHES_PER_EPOCH * batch_size]
            train_subset = torch.utils.data.Subset(dataset, train_indices)
            dataset= train_subset

    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=shuffle, 
                            num_workers=8, 
                            pin_memory=True,
                            prefetch_factor=2,
                            persistent_workers=True)

    return dataloader

# %%
def accuracy_fn(y_true, y_pred):
  correct = torch.eq(y_true, y_pred).sum().item()
  acc = (correct/len(y_pred)) * 100
  return acc

# %%
def train(model, train_loader, optimizer, loss_fn, metrics_handler, epoch, device):
    scaler = GradScaler('cuda')
    model.train()
    metrics_handler.reset("train")
    start_time = time.time()

    train_loss, train_acc = 0, 0

    for batch_idx, (X, y) in enumerate(tqdm(train_loader)):
        X, y = X.to(device), y.float().to(device)

        optimizer.zero_grad()
        with autocast(device_type='cuda'):
            pred_logits = model(X).squeeze()
            loss = loss_fn(pred_logits.view(-1), y)

        train_loss += loss.item() # accumulate train loss

        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        scaler.step(optimizer)
        scaler.update()

        y_pred = torch.round(torch.sigmoid(pred_logits))

        metrics_handler.update('train', y_pred, y, loss.item())

        train_acc += accuracy_fn(y, y_pred)

        if batch_idx % 100 == 0:

            torch.cuda.empty_cache()  # Clear CUDA cache to prevent memory leaks

    # Divide total train loss by length of train dataloader
    train_loss /= len(train_loader)
    train_acc /= len(train_loader)

    # Compute and log epoch metrics
    epoch_metrics = metrics_handler.log_epoch_metrics(epoch, 'train')
    epoch_time = time.time() - start_time
    print(f'Training epoch time: {epoch_time:.2f}s')

    return epoch_metrics

# %%
def test(model, test_loader, loss_fn, metrics_handler, epoch, device):
    test_loss, test_acc = 0, 0
    real_as_real, real_as_fake, fake_as_real, fake_as_fake = 0, 0, 0, 0
    model.eval()
    metrics_handler.reset('val')
    start_time = time.time()

    with torch.inference_mode():
        for i, (X_test, y_test) in enumerate(tqdm(test_loader)):
            X_test, y_test = X_test.to(device), y_test.float().to(device)

            test_logits = model(X_test).squeeze()
            test_pred = torch.round(torch.sigmoid(test_logits))
            test_loss += loss_fn(test_logits.view(-1), y_test.float()).item()
            
            test_acc += accuracy_fn(y_true=y_test, 
                                    y_pred=test_pred)

            # Update metrics
            metrics_handler.update('val', test_pred, y_test, test_loss)

            # Ensure y_test and test_pred are still on the GPU
            fake_as_fake += ((y_test == 0) & (test_pred == 0)).sum().item()
            fake_as_real += ((y_test == 0) & (test_pred == 1)).sum().item()
            real_as_fake += ((y_test == 1) & (test_pred == 0)).sum().item()
            real_as_real += ((y_test == 1) & (test_pred == 1)).sum().item()

        test_loss /= len(test_loader)
        test_acc /= len(test_loader)

        # Compute and log epoch metrics
        epoch_metrics = metrics_handler.log_epoch_metrics(epoch, 'val')
        epoch_time = time.time() - start_time
        print(f'Validation epoch time: {epoch_time:.2f}s')

        print(f"Real images identified as real: {real_as_real}")
        print(f"Real images identified as fake: {real_as_fake}")
        print(f"Fake images identified as real: {fake_as_real}")
        print(f"Fake images identified as fake: {fake_as_fake}")

    return epoch_metrics

# %%
def save_model(model):
    # Create saved_states directory if it doesn't exist
    os.makedirs("saved_states_b6", exist_ok=True)

    # Generate timestamp and filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"combined_model_b6_{timestamp}.pth"

    # Construct full path
    model_save_path = os.path.join("/home/nithira/sp_cup/saved_states_b6", model_filename)

    # Save model state dictionary
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

# %%
def load_model(model, model_path, device):
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    print("Model loaded successfully")

# %%
def train_model(model, train_dir, valid_loader, device,  metrics_handler, batch_size, train_transform,  early_stopping_patience, lr, epochs):
    # Setup loss function and optimizer
    loss_fn = nn.BCEWithLogitsLoss()

    # Define the optimizer
    optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=1e-5)

    # Sheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    # Initialize metric handler
    metric_handler = MetricsHandler(
        num_classes=model.num_classes if hasattr(model, 'num_classes') else 2,
        device=device,
        log_dir=f"runs/experiment_{time.strftime('%Y%m%d_%H%M%S')}"
    )

    # Create save directory
    save_dir = Path(f"/home/nithira/sp_cup/checkpoints/checkpoint_{time.strftime('%Y%m%d_%H%M%S')}")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Initialize history
    history = {
        'train_metrics': [],
        'val_metrics': [],
    }

    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}\n------")

        ### Training
        train_loader = get_data_loaders(train_dir, batch_size, transform=train_transform, shuffle=True)

        train_metrics = train(model, train_loader, optimizer, loss_fn, metrics_handler,epoch, device)
        history['train_metrics'].append(train_metrics)

        ### Testing
        val_metrics = test(model, valid_loader, loss_fn, metrics_handler,epoch, device)
        scheduler.step(val_metrics['loss'])
        history['val_metrics'].append(val_metrics)


        # Save checkpoint if best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            
            # Save best model
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
            }
            torch.save(checkpoint, save_dir / 'best_model.pth')
            print(f'Saved best model checkpoint to {save_dir/"best_model.pth"}')
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break

        if (epoch + 1) % 5 == 0:  # Save every 5 epochs
            save_model(model)

    
    # Close TensorBoard writer
    metric_handler.close()

    return history

# %% 
def main():

    # Configure for maximum GPU utilization
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    BATCH_SIZE = 32
    WIDTH = 128
    HEIGHT = 128

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")

    num_workers = multiprocessing.cpu_count()
    num_workers

    train_dir = "/home/nithira/sp_cup/dataset/train"
    valid_dir = "/home/nithira/sp_cup/dataset/valid"

    train_real = (os.listdir(f"{train_dir}/real"))
    train_fake = (os.listdir(f"{train_dir}/fake"))
    valid_real = (os.listdir(f"{valid_dir}/real"))
    valid_fake = (os.listdir(f"{valid_dir}/fake"))

    print(f"Training dataset size: {len(train_real) + len(train_fake)} (Real: {len(train_real)}, Fake: {len(train_fake)})")
    print(f"Validation dataset size: {len(valid_real) + len(valid_fake)} (Real: {len(valid_real)}, Fake: {len(valid_fake)})")

    training_mean, training_std = compute_mean_and_std(train_dir)
    validating_mean, validating_std = compute_mean_and_std(valid_dir)

    print(f"Training Mean: {training_mean}, Training Std: {training_std}")
    print(f"Validation Mean: {validating_mean}, Validation Std: {validating_std}")

    train_transform = transforms.Compose([
        transforms.Resize((HEIGHT, WIDTH)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=training_mean.tolist(), std=training_std.tolist())
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((HEIGHT, WIDTH)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=validating_mean.tolist(), std=validating_std.tolist())
    ])

    # Get data loaders
    train_loader = get_data_loaders(data_dir=train_dir,
                                    batch_size=BATCH_SIZE,
                                    transform=train_transform,
                                    balanced=True,
                                    shuffle=True)

    valid_loader = get_data_loaders(data_dir=valid_dir, 
                                    batch_size=128,
                                    balanced=False,
                                    transform=valid_transform)

    '''class_names = train_loader.dataset.classes
    print(f"Class names: {class_names}")
    class_to_idx = train_loader.dataset.class_to_idx
    print(f"Class to index: {class_to_idx}")'''

    print(f"Dataloader: {len(train_loader)} batches of {BATCH_SIZE} images")
    print(f"Dataloader: {len(valid_loader)} batches of {BATCH_SIZE} images") 

    model = CombinedModel(HEIGHT, WIDTH)
    model.to(device)
    print( next(model.parameters()).device)
    # print(model)

    metrics_handler = MetricsHandler()

    # Ensure all parameters require gradients
    """for param in model.parameters():
        param.requires_grad = False"""


    # Load the trained model
    loaded_model = CombinedModel(HEIGHT, WIDTH)
    load_model(loaded_model, "/home/nithira/sp_cup/saved_states/combined_model_b6_20250113_201930.pth", device)

    # Train the model
    history = train_model(loaded_model, train_dir, valid_loader, device, metrics_handler, BATCH_SIZE, train_transform, early_stopping_patience=100, lr=0.00025, epochs=20)

if __name__ == "__main__":
    main()
