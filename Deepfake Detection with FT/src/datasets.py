import os
import torch
import torchvision.utils
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from imageio import imread
from torchvision import datasets, transforms
from tqdm import tqdm
import multiprocessing
from PIL import Image
from skimage.io import imread
import matplotlib.pyplot as plt


"""class Dataset(TorchDataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = imread(self.image_paths[idx])
        label = self.labels[idx]

        if self.transform:
            image = Image.fromarray(image)
            image = self.transform(image)


        return image, label

def get_image_paths_and_labels(data_dir):
    image_paths = []
    labels = []
    class_names = os.listdir(data_dir)
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_dir):
            image_paths.append(os.path.join(class_dir, img_name))
            labels.append(class_to_idx[class_name])
    return image_paths, labels"""

def get_data_loaders(train_dir, test_dir, batch_size, transform=None):
    """
    Create train and test data loaders using torchvision.datasets.ImageFolder.
    """

    # Create datasets using ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)

    # Create DataLoaders for train and test datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Function to show a batch of images
def visualize_batch(images, labels, class_names=None):
    """
    Visualizes a batch of images with their labels.
    
    Args:
        images (torch.Tensor): Batch of images of shape (batch_size, channels, height, width).
        labels (torch.Tensor): Corresponding labels for the images.
        class_names (list): Optional list of class names corresponding to label indices.
    """
    batch_size = images.size(0)  # Get the number of images in the batch
    nrow = batch_size  # Set the number of images per row to match batch size
    
    # Convert images to a grid
    images_grid = torchvision.utils.make_grid(images, nrow=nrow, normalize=True).permute(1, 2, 0)
    
    # Prepare label titles
    if class_names:
        labels_str = [class_names[label] for label in labels]
    else:
        labels_str = [str(label.item()) for label in labels]
    
    # Plot the images
    plt.figure(figsize=(nrow * 2, 4))  # Adjust figure size based on the number of images
    plt.imshow(images_grid)
    plt.axis("off")
    plt.title("Labels: " + ", ".join(labels_str))
    plt.show()

# Compute mean and std of the dataset
def compute_mean_and_std(data_dir):
    """
    Compute per-channel mean and std of the dataset (to be used in transforms.Normalize())
    """

    cache_file = "logs/mean_and_std.pt"
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