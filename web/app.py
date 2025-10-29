"""
Flask web application for AI Shape Detector
Beautiful, minimal interface for shape detection
"""
import os
import time
import torch
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
from torchvision import transforms
from werkzeug.utils import secure_filename

# Import model classes
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import ShapeDetectorCNN
from model_advanced import SimpleAdvancedCNN, AdvancedShapeDetectorCNN

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'uploads')

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Global model variable
model = None
class_names = None
device = None


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model():
    """Load the trained model on startup"""
    global model, class_names, device

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Try to load advanced model first, fallback to basic model
    model_paths = [
        '../models/best_model_advanced.pth',
        '../models/best_model.pth',
        'models/best_model_advanced.pth',
        'models/best_model.pth'
    ]

    model_loaded = False
    for model_path in model_paths:
        full_path = os.path.join(os.path.dirname(__file__), model_path)
        if os.path.exists(full_path):
            try:
                print(f"Loading model from: {full_path}")
                checkpoint = torch.load(full_path, map_location=device)

                class_names = checkpoint.get('class_names', ['circle', 'rectangle', 'square', 'triangle'])
                model_type = checkpoint.get('model_type', 'basic')
                num_classes = len(class_names)

                # Create appropriate model
                if model_type == 'advanced':
                    model = AdvancedShapeDetectorCNN(num_classes=num_classes)
                elif model_type == 'simple':
                    model = SimpleAdvancedCNN(num_classes=num_classes)
                else:
                    model = ShapeDetectorCNN(num_classes=num_classes)

                model.load_state_dict(checkpoint['model_state_dict'])
                model = model.to(device)
                model.eval()

                print(f"✓ Model loaded successfully!")
                print(f"  Type: {model_type}")
                print(f"  Classes: {class_names}")
                if 'val_acc' in checkpoint:
                    print(f"  Validation Accuracy: {checkpoint['val_acc']:.2f}%")

                model_loaded = True
                break

            except Exception as e:
                print(f"Failed to load model from {full_path}: {e}")
                continue

    if not model_loaded:
        print("ERROR: Could not load any model!")
        print("Please ensure you have trained a model and it's saved in the models/ directory")
        class_names = ['circle', 'rectangle', 'square', 'triangle']

    # Ensure upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def preprocess_image(image_path):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor


@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device),
        'classes': class_names
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""

    if model is None:
        return jsonify({
            'error': 'Model not loaded',
            'message': 'Please train a model first'
        }), 500

    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, GIF, BMP'}), 400

    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess image
        start_time = time.time()
        image_tensor = preprocess_image(filepath)
        image_tensor = image_tensor.to(device)

        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        processing_time = time.time() - start_time

        # Get all probabilities
        all_probs = probabilities[0].cpu().numpy()
        all_predictions = {
            class_names[i]: float(prob * 100) for i, prob in enumerate(all_probs)
        }

        # Sort predictions by confidence
        all_predictions = dict(sorted(all_predictions.items(), key=lambda x: x[1], reverse=True))

        predicted_class = class_names[predicted.item()]
        confidence_score = float(confidence.item() * 100)

        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass

        return jsonify({
            'success': True,
            'prediction': predicted_class,
            'confidence': round(confidence_score, 2),
            'all_predictions': {k: round(v, 2) for k, v in all_predictions.items()},
            'processing_time': round(processing_time * 1000, 2)  # Convert to ms
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    print("="*60)
    print("🚀 AI Shape Detector - Web Interface")
    print("="*60)

    # Load model on startup
    load_model()

    print("\nStarting Flask server...")
    print("Open your browser and navigate to: http://localhost:5000")
    print("="*60)

    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
