"""
Simple CNN model for shape classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ShapeDetectorCNN(nn.Module):
    """
    Simple Convolutional Neural Network for classifying 4 basic shapes:
    circle, square, triangle, rectangle

    Architecture:
        - 3 Convolutional layers with increasing filters (32 -> 64 -> 128)
        - MaxPooling after each conv layer
        - 2 Fully connected layers with dropout
        - 4-class output with softmax
    """

    def __init__(self, num_classes=4):
        super(ShapeDetectorCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        # Fully connected layers
        # After 3 pooling layers: 128x128 -> 64x64 -> 32x32 -> 16x16
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)  # 128x128 -> 64x64

        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)  # 64x64 -> 32x32

        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)  # 32x32 -> 16x16

        # Flatten
        x = x.view(-1, 128 * 16 * 16)

        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_model(num_classes=4, device='cpu'):
    """
    Create and return the model

    Args:
        num_classes: Number of shape classes to classify
        device: Device to load model on ('cpu' or 'cuda')

    Returns:
        model: ShapeDetectorCNN model
    """
    model = ShapeDetectorCNN(num_classes=num_classes)
    model = model.to(device)

    print(f"Model created with {model.count_parameters():,} trainable parameters")
    return model


if __name__ == "__main__":
    # Test model creation
    model = get_model()
    print("\nModel Architecture:")
    print(model)

    # Test forward pass
    dummy_input = torch.randn(1, 3, 128, 128)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output logits: {output}")
