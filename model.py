# model.py

import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)  # Input: 1x28x28 → Output: 32x26x26
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3) # Output: 64x24x24
        self.pool = nn.MaxPool2d(2, 2)                # After pool: 64x12x12
        self.fc1 = nn.Linear(64*12*12, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 output classes (digits 0-9)

    def forward(self, x):
        x = F.relu(self.conv1(x))     # Conv + ReLU
        x = F.relu(self.conv2(x))     # Conv + ReLU
        x = self.pool(x)              # Max Pool
        x = x.view(-1, 64*12*12)      # Flatten
        x = F.relu(self.fc1(x))       # Fully Connected
        x = self.fc2(x)               # Output layer
        return x
