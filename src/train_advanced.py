"""
Advanced training script for robust shape detector
Features:
- Advanced data augmentation
- Deeper model architecture
- More epochs with early stopping
- Better learning rate scheduling
- Mixup augmentation (optional)
- Comprehensive logging
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from model_advanced import get_model
from augmentations import get_train_transforms, get_val_transforms, mixup_data, mixup_criterion


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def get_dataloaders(data_dir='data', batch_size=64, num_workers=0, use_augmentation=True):
    """
    Create data loaders for train and validation sets with advanced augmentation

    Args:
        data_dir: Root directory containing train/val folders
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        use_augmentation: Whether to use data augmentation

    Returns:
        train_loader, val_loader, class_names
    """
    # Get transforms
    if use_augmentation:
        train_transform = get_train_transforms(img_size=128)
    else:
        train_transform = get_val_transforms(img_size=128)

    val_transform = get_val_transforms(img_size=128)

    # Load datasets
    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'train'),
        transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'val'),
        transform=val_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    class_names = train_dataset.classes

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Classes: {class_names}")
    print(f"Batch size: {batch_size}")
    print(f"Augmentation: {use_augmentation}")

    return train_loader, val_loader, class_names


def train_one_epoch(model, train_loader, criterion, optimizer, device, use_mixup=False, mixup_alpha=0.2):
    """Train for one epoch with optional mixup"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc='Training')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        # Apply mixup if enabled
        if use_mixup and np.random.random() > 0.5:
            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, mixup_alpha)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Mixup loss
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss / (pbar.n + 1):.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{running_loss / (pbar.n + 1):.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc


def plot_training_history(history, save_path='results/training_history_advanced.png'):
    """Plot training and validation metrics"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Plot loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', marker='o', markersize=4)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', marker='o', markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', marker='o', markersize=4)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', marker='o', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    plt.close()


def train(
    epochs=50,
    batch_size=64,
    learning_rate=0.001,
    data_dir='data',
    save_dir='models',
    model_type='simple',
    use_augmentation=True,
    use_mixup=False,
    early_stopping_patience=15,
    weight_decay=1e-4
):
    """
    Main training function with all enhancements

    Args:
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        data_dir: Directory containing train/val data
        save_dir: Directory to save model checkpoints
        model_type: 'advanced' (with residual) or 'simple' (deeper CNN)
        use_augmentation: Whether to use data augmentation
        use_mixup: Whether to use mixup augmentation
        early_stopping_patience: Patience for early stopping
        weight_decay: L2 regularization strength
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    train_loader, val_loader, class_names = get_dataloaders(
        data_dir, batch_size, use_augmentation=use_augmentation
    )

    # Create model
    model = get_model(model_type=model_type, num_classes=len(class_names), device=device)

    # Loss and optimizer with weight decay
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for better generalization
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Learning rate schedulers
    scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=learning_rate * 0.01
    )

    # Early stopping
    early_stopping = EarlyStopping(patience=early_stopping_patience)

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }

    best_val_acc = 0.0
    start_time = time.time()

    print(f"\n{'='*70}")
    print(f"Starting training for up to {epochs} epochs...")
    print(f"Model: {model_type}")
    print(f"Augmentation: {use_augmentation}, Mixup: {use_mixup}")
    print(f"Learning rate: {learning_rate}, Weight decay: {weight_decay}")
    print(f"Early stopping patience: {early_stopping_patience}")
    print(f"{'='*70}\n")

    for epoch in range(epochs):
        print(f"Epoch [{epoch+1}/{epochs}]")
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.6f}")

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            use_mixup=use_mixup
        )
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(current_lr)

        # Update learning rate
        scheduler_plateau.step(val_acc)
        scheduler_cosine.step()

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(save_dir, 'best_model_advanced.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'class_names': class_names,
                'model_type': model_type
            }, model_path)
            print(f"  ✓ Best model saved (Val Acc: {val_acc:.2f}%)")

        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
            break

        print()

    # Training complete
    elapsed_time = time.time() - start_time
    print(f"{'='*70}")
    print(f"Training complete in {elapsed_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Total epochs trained: {len(history['train_loss'])}")
    print(f"{'='*70}\n")

    # Plot training history
    plot_training_history(history)

    # Save final model
    final_model_path = os.path.join(save_dir, 'final_model_advanced.pth')
    torch.save({
        'epoch': len(history['train_loss']),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'class_names': class_names,
        'model_type': model_type
    }, final_model_path)
    print(f"Final model saved to {final_model_path}")

    return history, best_val_acc


if __name__ == "__main__":
    # Train with all enhancements
    train(
        epochs=50,                      # More epochs with early stopping
        batch_size=64,                  # Larger batch size for GPU
        learning_rate=0.001,
        data_dir='data',
        save_dir='models',
        model_type='simple',            # Use 'advanced' for residual model
        use_augmentation=True,          # Enable data augmentation
        use_mixup=False,                # Optional: enable mixup (can slow training)
        early_stopping_patience=15,     # Stop if no improvement for 15 epochs
        weight_decay=1e-4               # L2 regularization
    )
