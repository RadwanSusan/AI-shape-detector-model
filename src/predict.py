"""
Inference script for shape detection on custom images
"""
import os
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

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

    return model, class_names


def predict_image(image_path, model, class_names, device='cpu'):
    """
    Predict shape in a single image

    Args:
        image_path: Path to image file
        model: Trained model
        class_names: List of class names
        device: Device to run inference on

    Returns:
        predicted_class, confidence
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to(device)

    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = class_names[predicted.item()]
    confidence = confidence.item() * 100

    return predicted_class, confidence


def visualize_prediction(image_path, predicted_class, confidence, save_path=None):
    """Visualize prediction on image"""
    img = Image.open(image_path)

    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis('off')

    # Add prediction text
    title = f'Predicted: {predicted_class.upper()}\nConfidence: {confidence:.2f}%'
    color = 'green' if confidence > 80 else 'orange' if confidence > 60 else 'red'
    plt.title(title, fontsize=16, fontweight='bold', color=color, pad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()

    plt.close()


def predict_batch(image_dir, model, class_names, device='cpu', save_results=True):
    """
    Predict shapes for all images in a directory

    Args:
        image_dir: Directory containing images
        model: Trained model
        class_names: List of class names
        device: Device to run inference on
        save_results: Whether to save visualization
    """
    # Get all image files
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions)]

    if not image_files:
        print(f"No images found in {image_dir}")
        return

    print(f"Found {len(image_files)} images. Processing...\n")

    results = []

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)

        try:
            predicted_class, confidence = predict_image(image_path, model, class_names, device)
            results.append({
                'file': image_file,
                'prediction': predicted_class,
                'confidence': confidence
            })

            print(f"{image_file:30s} -> {predicted_class:10s} ({confidence:.2f}%)")

        except Exception as e:
            print(f"Error processing {image_file}: {e}")

    # Save results
    if save_results and results:
        output_dir = 'results/predictions'
        os.makedirs(output_dir, exist_ok=True)

        # Create visualization for first few images
        num_viz = min(len(image_files), 6)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()

        for i in range(num_viz):
            image_path = os.path.join(image_dir, results[i]['file'])
            img = Image.open(image_path)

            axes[i].imshow(img)
            axes[i].axis('off')

            pred = results[i]['prediction']
            conf = results[i]['confidence']
            color = 'green' if conf > 80 else 'orange' if conf > 60 else 'red'
            axes[i].set_title(f'{pred}\n{conf:.1f}%', color=color, fontsize=12, fontweight='bold')

        plt.tight_layout()
        save_path = os.path.join(output_dir, 'batch_predictions.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nBatch visualization saved to {save_path}")
        plt.close()

    return results


def main():
    """Main prediction function"""
    import argparse

    parser = argparse.ArgumentParser(description='Predict shapes in images')
    parser.add_argument('--image', type=str, help='Path to single image file')
    parser.add_argument('--dir', type=str, help='Path to directory with images')
    parser.add_argument('--model', type=str, default='models/best_model.pth', help='Path to model checkpoint')
    parser.add_argument('--save', action='store_true', help='Save visualization')

    args = parser.parse_args()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.model}...")
    model, class_names = load_model(args.model, device)
    print(f"Model loaded. Classes: {class_names}\n")

    if args.image:
        # Predict single image
        if not os.path.exists(args.image):
            print(f"Error: Image not found at {args.image}")
            return

        print(f"Predicting shape in {args.image}...")
        predicted_class, confidence = predict_image(args.image, model, class_names, device)

        print(f"\nPrediction: {predicted_class.upper()}")
        print(f"Confidence: {confidence:.2f}%")

        if args.save:
            save_path = f'results/prediction_{os.path.basename(args.image)}'
            visualize_prediction(args.image, predicted_class, confidence, save_path)
        else:
            visualize_prediction(args.image, predicted_class, confidence)

    elif args.dir:
        # Predict batch of images
        if not os.path.exists(args.dir):
            print(f"Error: Directory not found at {args.dir}")
            return

        predict_batch(args.dir, model, class_names, device, save_results=args.save)

    else:
        print("Error: Please provide either --image or --dir argument")
        parser.print_help()


if __name__ == "__main__":
    # If no arguments provided, show help
    import sys
    if len(sys.argv) == 1:
        print("Shape Detector - Predict shapes in images\n")
        print("Usage examples:")
        print("  python predict.py --image path/to/image.png")
        print("  python predict.py --dir path/to/images --save")
        print("  python predict.py --image test.png --model models/best_model.pth --save")
        print("\nFor more options, use: python predict.py --help")
    else:
        main()
