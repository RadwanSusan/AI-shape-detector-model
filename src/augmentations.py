"""
Advanced data augmentation transforms for training
"""
import torch
import torchvision.transforms as transforms
import random


class RandomErasing:
    """Randomly erase a rectangular region in the image"""
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.random() > self.probability:
            return img

        img_tensor = transforms.ToTensor()(img)
        _, h, w = img_tensor.shape
        area = h * w

        for _ in range(100):
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h_erase = int(round((target_area * aspect_ratio) ** 0.5))
            w_erase = int(round((target_area / aspect_ratio) ** 0.5))

            if h_erase < h and w_erase < w:
                i = random.randint(0, h - h_erase)
                j = random.randint(0, w - w_erase)
                img_tensor[:, i:i + h_erase, j:j + w_erase] = 0
                return transforms.ToPILImage()(img_tensor)

        return img


def get_train_transforms(img_size=128):
    """
    Get training data augmentation transforms

    Returns robust augmentations for shape detection
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        # Geometric transforms
        transforms.RandomRotation(degrees=15, fill=0),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),

        # Color jittering for robustness to lighting variations
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1
        ),

        # Random affine for slight distortions
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            fill=0
        ),

        # Convert to tensor
        transforms.ToTensor(),

        # Normalization
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),

        # Random erasing (after normalization)
        transforms.RandomErasing(
            p=0.3,
            scale=(0.02, 0.15),
            ratio=(0.3, 3.3),
            value=0
        )
    ])


def get_val_transforms(img_size=128):
    """
    Get validation/test data transforms (no augmentation)
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def mixup_data(x, y, alpha=0.2):
    """
    Mixup: Beyond Empirical Risk Minimization
    Mixes two images and their labels

    Args:
        x: Input images batch
        y: Target labels batch
        alpha: Mixup interpolation strength

    Returns:
        mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Mixup loss function

    Args:
        criterion: Base loss function
        pred: Model predictions
        y_a: Original labels
        y_b: Mixed labels
        lam: Mixup lambda value

    Returns:
        Mixed loss value
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# For mixup
import numpy as np


class MixUpWrapper:
    """
    Wrapper class for applying mixup augmentation
    """
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, x, y):
        return mixup_data(x, y, self.alpha)

    def loss(self, criterion, pred, y_a, y_b, lam):
        return mixup_criterion(criterion, pred, y_a, y_b, lam)


if __name__ == "__main__":
    # Test augmentations
    from PIL import Image
    import numpy as np

    # Create a simple test image
    test_img = Image.new('RGB', (128, 128), color='red')

    # Test training transforms
    train_transforms = get_train_transforms()
    augmented = train_transforms(test_img)
    print(f"Training transform output shape: {augmented.shape}")
    print(f"Min value: {augmented.min():.3f}, Max value: {augmented.max():.3f}")

    # Test validation transforms
    val_transforms = get_val_transforms()
    val_img = val_transforms(test_img)
    print(f"Validation transform output shape: {val_img.shape}")

    print("\n✓ Augmentations working correctly!")
