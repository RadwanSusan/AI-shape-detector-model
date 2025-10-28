"""
Evaluation script for shape detector model
"""
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from model import ShapeDetectorCNN


def load_model(model_path, device='cpu'):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(model_path, map_location=device)

    # Get class names from checkpoint
    class_names = checkpoint.get('class_names', ['circle', 'rectangle', 'square', 'triangle'])
    num_classes = len(class_names)

    # Create model and load weights
    model = ShapeDetectorCNN(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded from {model_path}")
    print(f"Classes: {class_names}")

    return model, class_names


def get_test_loader(data_dir='data', batch_size=32):
    """Create test data loader"""
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    test_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'test'),
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"Test samples: {len(test_dataset)}")
    return test_loader, test_dataset.classes


def evaluate_model(model, test_loader, device):
    """
    Evaluate model on test set

    Returns:
        accuracy, all_labels, all_predictions
    """
    model.eval()
    all_predictions = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Evaluating')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            # Collect predictions
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Calculate accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({'acc': f'{100 * correct / total:.2f}%'})

    accuracy = 100 * correct / total
    return accuracy, np.array(all_labels), np.array(all_predictions)


def plot_confusion_matrix(cm, class_names, save_path='results/confusion_matrix.png'):
    """Plot confusion matrix"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_sample_predictions(model, test_loader, class_names, device, num_samples=16):
    """Plot sample predictions"""
    model.eval()

    # Get one batch
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)

    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)

    # Denormalize images for display
    images = images.cpu()
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    images = images * std + mean
    images = torch.clamp(images, 0, 1)

    # Plot
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()

    for i in range(min(num_samples, len(images))):
        ax = axes[i]
        img = images[i].permute(1, 2, 0).numpy()
        ax.imshow(img)

        true_label = class_names[labels[i]]
        pred_label = class_names[predictions[i]]
        color = 'green' if true_label == pred_label else 'red'

        ax.set_title(f'True: {true_label}\nPred: {pred_label}', color=color, fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    save_path = 'results/sample_predictions.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Sample predictions saved to {save_path}")
    plt.close()


def main(model_path='models/best_model.pth', data_dir='data'):
    """Main evaluation function"""
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load model
    model, class_names = load_model(model_path, device)

    # Load test data
    test_loader, test_class_names = get_test_loader(data_dir)

    # Evaluate
    print(f"\n{'='*60}")
    print("Evaluating model on test set...")
    print(f"{'='*60}\n")

    accuracy, all_labels, all_predictions = evaluate_model(model, test_loader, device)

    # Print results
    print(f"\n{'='*60}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"{'='*60}\n")

    # Classification report
    print("Classification Report:")
    print("="*60)
    report = classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        digits=4
    )
    print(report)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    print("\nConfusion Matrix:")
    print(cm)
    plot_confusion_matrix(cm, class_names)

    # Plot sample predictions
    print("\nGenerating sample predictions...")
    plot_sample_predictions(model, test_loader, class_names, device)

    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main(model_path='models/best_model.pth', data_dir='data')
