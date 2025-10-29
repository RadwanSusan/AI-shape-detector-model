"""
Advanced CNN model for robust shape classification
Features:
- Deeper architecture (5 conv layers)
- Residual connections for better gradient flow
- More filters for better feature extraction
- Better regularization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection
    Helps with training deeper networks
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residual)
        out = F.relu(out)

        return out


class AdvancedShapeDetectorCNN(nn.Module):
    """
    Advanced CNN for shape classification with residual connections

    Architecture:
        - 5 Convolutional blocks with increasing filters (32→64→128→256→512)
        - Residual connections for better gradient flow
        - Batch normalization after each conv layer
        - Dropout for regularization
        - Global Average Pooling before FC layers
        - 2 Fully connected layers with dropout

    Input: 128x128 RGB images
    Output: 4 classes (circle, square, triangle, rectangle)
    """

    def __init__(self, num_classes=4, dropout_rate=0.6):
        super(AdvancedShapeDetectorCNN, self).__init__()

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # Residual blocks with increasing channels
        self.layer1 = self._make_layer(32, 64, stride=2)   # 128→64
        self.layer2 = self._make_layer(64, 128, stride=2)  # 64→32
        self.layer3 = self._make_layer(128, 256, stride=2) # 32→16
        self.layer4 = self._make_layer(256, 512, stride=2) # 16→8

        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout_rate * 0.7)

        self.fc3 = nn.Linear(128, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, stride):
        """Create a residual layer"""
        return ResidualBlock(in_channels, out_channels, stride)

    def _initialize_weights(self):
        """Initialize network weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)

        return x

    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SimpleAdvancedCNN(nn.Module):
    """
    Simpler advanced CNN without residual connections
    Good balance between complexity and performance
    """
    def __init__(self, num_classes=4, dropout_rate=0.6):
        super(SimpleAdvancedCNN, self).__init__()

        # Convolutional layers - deeper architecture
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)

        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        # After 5 pooling layers: 128 -> 64 -> 32 -> 16 -> 8 -> 4
        self.fc1 = nn.Linear(512 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)  # 128 -> 64

        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)  # 64 -> 32

        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)  # 32 -> 16

        # Conv block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool(x)  # 16 -> 8

        # Conv block 5
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.pool(x)  # 8 -> 4

        # Flatten
        x = x.view(-1, 512 * 4 * 4)

        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)

        return x

    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_model(model_type='advanced', num_classes=4, device='cpu'):
    """
    Create and return the model

    Args:
        model_type: 'advanced' (with residual) or 'simple' (without residual)
        num_classes: Number of shape classes to classify
        device: Device to load model on ('cpu' or 'cuda')

    Returns:
        model: Shape detector model
    """
    if model_type == 'advanced':
        model = AdvancedShapeDetectorCNN(num_classes=num_classes)
    else:
        model = SimpleAdvancedCNN(num_classes=num_classes)

    model = model.to(device)

    print(f"Model type: {model_type}")
    print(f"Model created with {model.count_parameters():,} trainable parameters")
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing Advanced Model (with residual connections):")
    model_advanced = get_model('advanced')
    print(model_advanced)

    print("\n" + "="*60 + "\n")

    print("Testing Simple Advanced Model (without residual):")
    model_simple = get_model('simple')
    print(model_simple)

    # Test forward pass
    dummy_input = torch.randn(2, 3, 128, 128)

    print("\n" + "="*60 + "\n")
    print("Testing forward pass:")
    output_advanced = model_advanced(dummy_input)
    output_simple = model_simple(dummy_input)

    print(f"Advanced model - Input shape: {dummy_input.shape}")
    print(f"Advanced model - Output shape: {output_advanced.shape}")
    print(f"Advanced model - Output logits example: {output_advanced[0]}")

    print(f"\nSimple model - Input shape: {dummy_input.shape}")
    print(f"Simple model - Output shape: {output_simple.shape}")
    print(f"Simple model - Output logits example: {output_simple[0]}")
