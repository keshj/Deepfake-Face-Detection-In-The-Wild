import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models import Model
from datasets import Dataset
from train import train
from test import evaluate
from tensorboard_utils import setup_tensorboard

def parse_args():
    parser = argparse.ArgumentParser(description="Start training the nural network")
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory for TensorBoard logs')

    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup TensorBoard
    writer = setup_tensorboard(args.log_dir)

    # Initialize model
    model = Model()
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Load datasets
    train_dataset = Dataset(train=True)
    test_dataset = Dataset(train=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Train the model
    train(model, train_loader, criterion, optimizer, args.epochs, writer, args.checkpoint_dir)

    # Evaluate the model
    evaluate(model, test_loader, writer)

if __name__ == "__main__":
    main()
