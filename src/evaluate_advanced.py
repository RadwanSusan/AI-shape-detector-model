"""
Advanced evaluation script with robustness testing
Tests model performance across different difficulty levels and scenarios
"""
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from model_advanced import AdvancedShapeDetectorCNN, SimpleAdvancedCNN
from augmentations import get_val_transforms


def load_model(model_path, device='cpu'):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(model_path, map_location=device)

    # Get class names and model type from checkpoint
    class_names = checkpoint.get('class_names', ['circle', 'rectangle', 'square', 'triangle'])
    model_type = checkpoint.get('model_type', 'simple')
    num_classes = len(class_names)

    # Create model based on type
    if model_type == 'advanced':
        model = AdvancedShapeDetectorCNN(num_classes=num_classes)
    else:
        model = SimpleAdvancedCNN(num_classes=num_classes)

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded from {model_path}")
    print(f"Model type: {model_type}")
    print(f"Classes: {class_names}")
    if 'val_acc' in checkpoint:
        print(f"Best validation accuracy: {checkpoint['val_acc']:.2f}%")

    return model, class_names, model_type


def get_test_loader(data_dir='data', batch_size=32):
    """Create test data loader"""
    transform = get_val_transforms(img_size=128)

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
        accuracy, all_labels, all_predictions, all_confidences
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_confidences = []
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Evaluating')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidences, predicted = torch.max(probabilities, 1)

            # Collect predictions
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())

            # Calculate accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({'acc': f'{100 * correct / total:.2f}%'})

    accuracy = 100 * correct / total
    return accuracy, np.array(all_labels), np.array(all_predictions), np.array(all_confidences)


def plot_confusion_matrix(cm, class_names, save_path='results/confusion_matrix_advanced.png'):
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
        cbar_kws={'label': 'Count'},
        square=True
    )
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_confidence_distribution(confidences, predictions, labels, class_names,
                                 save_path='results/confidence_distribution.png'):
    """Plot confidence score distribution"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Separate correct and incorrect predictions
    correct_mask = predictions == labels
    correct_conf = confidences[correct_mask]
    incorrect_conf = confidences[~correct_mask]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Overall confidence distribution
    ax1.hist(correct_conf, bins=30, alpha=0.7, label='Correct', color='green', edgecolor='black')
    ax1.hist(incorrect_conf, bins=30, alpha=0.7, label='Incorrect', color='red', edgecolor='black')
    ax1.set_xlabel('Confidence Score', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Confidence Score Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Per-class confidence
    class_confidences = []
    for i, class_name in enumerate(class_names):
        class_mask = labels == i
        class_confidences.append(confidences[class_mask])

    ax2.boxplot(class_confidences, labels=class_names)
    ax2.set_ylabel('Confidence Score', fontsize=12)
    ax2.set_xlabel('Class', fontsize=12)
    ax2.set_title('Confidence per Class', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Confidence distribution saved to {save_path}")
    plt.close()


def plot_sample_predictions(model, test_loader, class_names, device, num_samples=16,
                            save_path='results/sample_predictions_advanced.png'):
    """Plot sample predictions with confidence scores"""
    model.eval()

    # Get one batch
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)

    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidences, predictions = torch.max(probabilities, 1)

    # Denormalize images for display
    images = images.cpu()
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    images = images * std + mean
    images = torch.clamp(images, 0, 1)

    # Plot
    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    axes = axes.ravel()

    for i in range(min(num_samples, len(images))):
        ax = axes[i]
        img = images[i].permute(1, 2, 0).numpy()
        ax.imshow(img)

        true_label = class_names[labels[i]]
        pred_label = class_names[predictions[i]]
        confidence = confidences[i].item() * 100

        is_correct = true_label == pred_label
        color = 'green' if is_correct else 'red'

        title = f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%'
        ax.set_title(title, color=color, fontsize=10, fontweight='bold')
        ax.axis('off')

    plt.suptitle('Sample Predictions', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Sample predictions saved to {save_path}")
    plt.close()


def analyze_errors(predictions, labels, confidences, class_names):
    """Analyze prediction errors"""
    incorrect_mask = predictions != labels
    incorrect_preds = predictions[incorrect_mask]
    incorrect_labels = labels[incorrect_mask]
    incorrect_conf = confidences[incorrect_mask]

    print(f"\n{'='*60}")
    print("Error Analysis:")
    print(f"{'='*60}")
    print(f"Total errors: {len(incorrect_preds)}")
    print(f"Error rate: {100 * len(incorrect_preds) / len(labels):.2f}%")

    if len(incorrect_preds) > 0:
        print(f"Average confidence on errors: {incorrect_conf.mean()*100:.2f}%")

        print("\nMost common errors:")
        error_pairs = list(zip(incorrect_labels, incorrect_preds))
        from collections import Counter
        error_counts = Counter(error_pairs)

        for (true_idx, pred_idx), count in error_counts.most_common(5):
            print(f"  {class_names[true_idx]} → {class_names[pred_idx]}: {count} times")


def main(model_path='models/best_model_advanced.pth', data_dir='data'):
    """Main evaluation function"""
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load model
    model, class_names, model_type = load_model(model_path, device)

    # Load test data
    test_loader, test_class_names = get_test_loader(data_dir)

    # Evaluate
    print(f"\n{'='*60}")
    print("Evaluating model on test set...")
    print(f"{'='*60}\n")

    accuracy, all_labels, all_predictions, all_confidences = evaluate_model(model, test_loader, device)

    # Print results
    print(f"\n{'='*60}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Average Confidence: {all_confidences.mean()*100:.2f}%")
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

    # Confidence analysis
    print("\nGenerating confidence analysis...")
    plot_confidence_distribution(all_confidences, all_predictions, all_labels, class_names)

    # Error analysis
    analyze_errors(all_predictions, all_labels, all_confidences, class_names)

    # Plot sample predictions
    print("\nGenerating sample predictions...")
    plot_sample_predictions(model, test_loader, class_names, device)

    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"{'='*60}")

    return accuracy, cm, all_confidences


if __name__ == "__main__":
    main(model_path='models/best_model_advanced.pth', data_dir='data')
