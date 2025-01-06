import os
import torch
from torch.utils.data import Dataset as TorchDataset
from imageio import imread
from torchvision import datasets, transforms
from tqdm import tqdm
import multiprocessing

class Dataset(torch.utils.data.Dataset):
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
            image = self.transform(image)

        return image, label


def get_data_loaders(train_image_paths, train_labels, test_image_paths, test_labels, batch_size, transform=None):
    train_dataset = Dataset(train_image_paths, train_labels, transform)
    valid_dataset = Dataset(test_image_paths, test_labels, transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader

# Compute mean and std of the dataset
def compute_mean_and_std(data_dir):
    """
    Compute per-channel mean and std of the dataset (to be used in transforms.Normalize())
    """
    cache_file = "mean_and_std.pt"
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

    mean = torch.zeros(3)
    var = torch.zeros(3)
    npix = 0

    for images, _ in tqdm(dl, total=len(ds), desc="Computing mean and std", ncols=80):
        images = images.view(3, -1)
        npix += images.size(1)
        mean += images.mean(1)
        var += images.var(1, unbiased=False)

    mean /= len(ds)
    std = torch.sqrt(var / len(ds))

    # Cache results so we don't need to redo the computation
    torch.save({"mean": mean, "std": std}, cache_file)

    return mean, std