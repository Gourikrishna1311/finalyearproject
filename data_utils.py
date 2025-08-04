# data_utils.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random

def load_mnist_dataset():
    transform = transforms.Compose([transforms.ToTensor()])

    train_set = datasets.MNIST(
        root='./dataset', train=True, download=True, transform=transform
    )
    test_set = datasets.MNIST(
        root='./dataset', train=False, download=True, transform=transform
    )

    return train_set, test_set

def partition_dataset(dataset, num_clients=3, seed=42):
    random.seed(seed)
    total_samples = len(dataset)
    indices = list(range(total_samples))
    random.shuffle(indices)

    # Equal split for now
    split_size = total_samples // num_clients
    client_indices = [indices[i*split_size : (i+1)*split_size] for i in range(num_clients)]

    # Handle leftover samples (add to last client)
    leftover = indices[num_clients*split_size:]
    if leftover:
        client_indices[-1].extend(leftover)

    client_subsets = [Subset(dataset, idxs) for idxs in client_indices]
    return client_subsets


# ----------------------------
# Create DataLoaders per Client
# ----------------------------
def create_client_loaders(subsets, batch_size=32):
    loaders = [DataLoader(subset, batch_size=batch_size, shuffle=True) for subset in subsets]
    return loaders
