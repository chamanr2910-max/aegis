import hashlib
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """
    A small Convolutional Neural Network for MNIST classification with 
    explicitly named layers: conv1, conv2, fc1, fc2 to enable modular layer hashing.
    """
    def __init__(self):
        super(Net, self).__init__()
        # Explicit named layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gracefully handle 3D inputs (N, 28, 28) from raw test tensors
        if x.dim() == 3:
            x = x.unsqueeze(1)
        # Convert uint8 inputs [0, 255] to float [0.0, 1.0] if necessary
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_layer(self, name: str) -> nn.Module:
        """Retrieve a layer by name (e.g. 'conv1', 'conv2', 'fc1', 'fc2')."""
        return getattr(self, name)

    def get_layers(self) -> Dict[str, nn.Module]:
        """Return a mapping of the named target layers."""
        return {
            "conv1": self.conv1,
            "conv2": self.conv2,
            "fc1": self.fc1,
            "fc2": self.fc2,
        }

    def hash_layer(self, name: str, algo: str = "sha256") -> str:
        """
        Helper method to compute the cryptographic hash of a specific named layer's weights and biases.
        Enables verification of layer integrity.
        """
        layer = self.get_layer(name)
        hasher = hashlib.new(algo)
        for param in layer.parameters():
            hasher.update(param.detach().cpu().numpy().tobytes())
        return hasher.hexdigest()


# Aliases for convenience across modules
CNN = Net
SmallCNN = Net
MNISTCNN = Net
Model = Net


if __name__ == "__main__":
    model = Net()
    print("Model Architecture:")
    print(model)

    dummy_input = torch.randn(2, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape for dummy input: {output.shape}")

    print("\nLayer Hashes (before training):")
    for name in ["conv1", "conv2", "fc1", "fc2"]:
        print(f"  {name}: {model.hash_layer(name)}")
