import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from models import Model
from datasets import get_data_loaders
from train import train
from test import evaluate, load_model
import torchvision.models as models

#from tensorboard_utils import setup_tensorboard

def parse_args():
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory for TensorBoard logs')
    parser.add_argument('--train_dir', type=str, default='D:/sp_cup/dataset/train', help='Directory for training data')
    parser.add_argument('--test_dir', type=str, default='D:/sp_cup/dataset/valid', help='Directory for testing data')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs to train')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup TensorBoard
    # writer = setup_tensorboard(args.log_dir)

    # Initialize model
    print("Initializing Model")

    #model = Model()

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_features = model.fc.in_features  # Get the number of features from the current fc layer
    model.fc = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 1), # Output layer for binary classification (Fake/Real)
    nn.Sigmoid()
)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Load datasets
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
        #transforms.Normalize([0.3996, 0.3194, 0.3223], [0.2321, 0.1766, 0.1816])
    ])
    train_loader, valid_loader = get_data_loaders(args.train_dir, args.test_dir, args.batch_size, transform)

    # Train the model
    print("Start Training")
    train(model, train_loader, criterion, optimizer, args.epochs, args.checkpoint_dir)

    model = load_model('/checkpoint.pth')
    # Evaluate the model
    print("Start Testing")
    evaluate(model, valid_loader)

if __name__ == "__main__":
    main()
