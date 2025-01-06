"""
This file contains functions for logging metrics and visualizations to TensorBoard.
"""

import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

class ImageLabelingLogger:
    """ Logger for logging images and their predicted labels to TensorBoard. """

    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def log_images(self, images, labels, predictions, step):
        """ Log images with their true and predicted labels. """
        for i in range(len(images)):
            self.writer.add_image(f'Image/{i}', images[i], step)
            self.writer.add_text(f'Label/{i}', f'True: {labels[i]}, Pred: {predictions[i]}', step)

    def close(self):
        self.writer.close()


class ConfusionMatrixLogger:
    """ Logger for logging confusion matrix to TensorBoard. """

    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def log_confusion_matrix(self, confusion_matrix, step):
        """ Log confusion matrix to TensorBoard. """
        self.writer.add_image('Confusion Matrix', self._plot_confusion_matrix(confusion_matrix), step)

    def _plot_confusion_matrix(self, cm):
        """ Create a plot of the confusion matrix. """
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()

        # Save to a numpy array
        plt.savefig('confusion_matrix.png')
        plt.close()
        return 'confusion_matrix.png'

    def close(self):
        self.writer.close()


class CustomModelSaver:
    """ Custom callback for saving model weights. """

    def __init__(self, checkpoint_dir, max_num_weights=5):
        self.checkpoint_dir = checkpoint_dir
        self.max_num_weights = max_num_weights
        self.checkpoints = []

    def save_model(self, model, epoch):
        """ Save the model weights. """
        if len(self.checkpoints) >= self.max_num_weights:
            os.remove(self.checkpoints.pop(0))
        checkpoint_path = os.path.join(self.checkpoint_dir, f'model_epoch_{epoch}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        self.checkpoints.append(checkpoint_path)
