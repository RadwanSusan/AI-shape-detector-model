# AI Shape Detector

A simple and minimal deep learning model built with PyTorch to detect basic geometric shapes (circle, square, triangle, rectangle) from images.

## Features

- **Simple CNN Architecture**: Lightweight 3-layer convolutional neural network (~100K parameters)
- **Synthetic Data Generation**: Automated generation of training data with variations
- **High Accuracy**: Achieves 95%+ accuracy on synthetic test data
- **Easy to Use**: Simple scripts for training, evaluation, and inference
- **🌐 Web Interface**: Beautiful, modern web app for easy shape detection
- **Educational**: Perfect for learning deep learning and computer vision fundamentals

## Project Structure

```
AI-shape-detector-model/
├── data/                   # Generated datasets (train/val/test)
├── src/
│   ├── generate_data.py   # Synthetic data generation
│   ├── model.py           # CNN architecture
│   ├── train.py           # Training script
│   ├── evaluate.py        # Model evaluation
│   └── predict.py         # Inference on new images
├── web/                   # 🌐 Web interface
│   ├── app.py            # Flask backend
│   ├── templates/        # HTML templates
│   ├── static/           # CSS, JS, assets
│   └── README.md         # Web interface docs
├── models/                # Saved model checkpoints
├── results/               # Training plots and metrics
├── requirements.txt       # Python dependencies
└── README.md
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or download this repository**

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

The required packages are:
- torch (PyTorch)
- torchvision
- Pillow (PIL)
- numpy
- matplotlib
- tqdm
- scikit-learn

## Usage

### Step 1: Generate Training Data

Generate synthetic shape images for training:

```bash
cd src
python generate_data.py
```

This will create 6,000 images (1,500 per shape class):
- **Train**: 4,200 images (70%)
- **Validation**: 900 images (15%)
- **Test**: 900 images (15%)

Generation takes ~2-5 minutes depending on your system.

### Step 2: Train the Model

Train the CNN model on the generated data:

```bash
python train.py
```

**Training parameters:**
- Epochs: 20
- Batch size: 32
- Learning rate: 0.001
- Optimizer: Adam
- Loss: Cross-entropy

**Expected training time:**
- CPU: 5-15 minutes
- GPU: 1-3 minutes

The best model (highest validation accuracy) is saved to `models/best_model.pth`.

### Step 3: Evaluate the Model

Evaluate the trained model on the test set:

```bash
python evaluate.py
```

This will:
- Calculate test accuracy and metrics
- Generate confusion matrix
- Show sample predictions
- Save visualizations to `results/`

**Expected accuracy:** 95%+ on synthetic test data

### Step 4: Predict on New Images

Use the trained model to predict shapes in your own images:

**Single image:**
```bash
python predict.py --image path/to/your/image.png
```

**Batch of images:**
```bash
python predict.py --dir path/to/images/folder --save
```

**With custom model:**
```bash
python predict.py --image test.png --model models/best_model.pth --save
```

## 🌐 Web Interface

Try the beautiful web interface for an easy-to-use experience!

### Quick Start

```bash
cd web
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:5000** in your browser.

### Features

- 🎨 **Modern, Minimal Design** - Clean and professional interface
- 📤 **Drag & Drop Upload** - Easy image uploading
- ⚡ **Real-Time Predictions** - Instant AI analysis
- 📊 **Confidence Scores** - Visual feedback with all class probabilities
- 📱 **Responsive** - Works on mobile, tablet, and desktop

### Screenshots

**Upload Interface:**
- Drag & drop your image
- Click to browse files
- Supports PNG, JPG, JPEG, GIF, BMP

**Results Display:**
- Large predicted shape name
- Confidence score with color coding
- Bar chart showing all predictions
- Processing time

For more details, see [web/README.md](web/README.md)

## Model Architecture

```
ShapeDetectorCNN
├── Conv2d (3 -> 32, 3x3) + BatchNorm + ReLU + MaxPool
├── Conv2d (32 -> 64, 3x3) + BatchNorm + ReLU + MaxPool
├── Conv2d (64 -> 128, 3x3) + BatchNorm + ReLU + MaxPool
├── Flatten
├── Linear (128*16*16 -> 256) + ReLU + Dropout(0.5)
└── Linear (256 -> 4)
```

**Total parameters:** ~32,768,388 (32M)

**Input:** 128x128 RGB images
**Output:** 4 classes (circle, square, triangle, rectangle)

## Customization

### Modify Shape Classes

Edit `src/generate_data.py`:
```python
SHAPES = ['circle', 'square', 'triangle', 'rectangle', 'pentagon']
```

### Adjust Training Parameters

Edit `src/train.py`:
```python
train(
    epochs=30,           # More epochs
    batch_size=64,       # Larger batch size
    learning_rate=0.0005 # Different learning rate
)
```

### Change Image Size

Edit both `src/generate_data.py` and `src/model.py`:
```python
IMG_SIZE = 256  # Higher resolution
```

## Results

### Expected Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | 95-99% |
| Training Time (CPU) | 5-15 min |
| Training Time (GPU) | 1-3 min |
| Model Size | ~125 MB |
| Inference Time | <10ms per image |

### Sample Output

```
Epoch [20/20]
  Train Loss: 0.0234 | Train Acc: 99.12%
  Val Loss:   0.0456 | Val Acc:   98.34%

Test Accuracy: 97.56%

Classification Report:
              precision    recall  f1-score   support
      circle     0.9789    0.9867    0.9828       225
   rectangle     0.9733    0.9644    0.9689       225
      square     0.9778    0.9778    0.9778       225
    triangle     0.9733    0.9778    0.9756       225
```

## Troubleshooting

### CUDA Out of Memory
Reduce batch size in `train.py`:
```python
batch_size=16  # or 8
```

### Low Accuracy
- Generate more training data (increase samples per class)
- Train for more epochs
- Add data augmentation
- Check if test data distribution matches training data

### Import Errors
Ensure all dependencies are installed:
```bash
pip install -r requirements.txt --upgrade
```

## Learning Resources

This project demonstrates:
- **CNNs**: How convolutional neural networks extract features
- **Data Generation**: Creating synthetic datasets programmatically
- **Training Loops**: Implementing training and validation cycles
- **Evaluation Metrics**: Understanding accuracy, precision, recall
- **Model Persistence**: Saving and loading trained models
- **Inference**: Using models for predictions on new data

## Extensions & Ideas

- Add more shape classes (pentagon, hexagon, star, etc.)
- Implement data augmentation (rotation, scaling, noise)
- Try transfer learning with pre-trained models (ResNet, VGG)
- Detect multiple shapes in one image (object detection)
- Deploy as a web API using Flask or FastAPI
- Create a GUI application with Tkinter or PyQt
- Train on real-world photos instead of synthetic data

## License

This project is open source and available for educational purposes.

## Contributing

Feel free to fork, modify, and improve this project. Suggestions and contributions are welcome!

## Acknowledgments

Built with:
- PyTorch (Deep Learning Framework)
- torchvision (Computer Vision Utilities)
- Pillow (Image Processing)

---

**Happy Learning!** 🎓

For questions or issues, please open an issue on GitHub.
