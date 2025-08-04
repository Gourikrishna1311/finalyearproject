# evaluate.py

import torch

def evaluate_model(model, dataloader, device='cpu'):
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():  # No gradient calculation during evaluation
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total * 100
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy
