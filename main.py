# main.py

from data_utils import load_mnist_dataset, partition_dataset, create_client_loaders
from model import CNN
from train_local_module import train_local
from evaluate import evaluate_model
from torch.utils.data import DataLoader

# Step 1: Load and partition dataset
train_set, test_set = load_mnist_dataset()
client_subsets = partition_dataset(train_set, num_clients=3)
client_loaders = create_client_loaders(client_subsets, batch_size=32)

# Test set loader
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

# Step 2: Train on one client
client_loader = client_loaders[0]
model = CNN()
trained_model = train_local(model, client_loader, epochs=5, lr=0.01, device='cpu')

# Step 3: Evaluate on the test set
evaluate_model(trained_model, test_loader, device='cpu')
