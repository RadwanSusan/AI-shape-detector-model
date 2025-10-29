# 🌐 AI Shape Detector - Web Interface

Beautiful, minimal web interface for the AI Shape Detector model.

## Features

✨ **Modern & Minimal Design**
- Clean, spacious layout with gradients
- Smooth animations and transitions
- Responsive (mobile, tablet, desktop)
- Professional aesthetic

🚀 **User-Friendly**
- Drag & drop image upload
- Real-time predictions
- Confidence scores with visual feedback
- All class probabilities displayed

⚡ **Fast Performance**
- Predictions in <100ms
- GPU support
- Efficient image processing

## Quick Start

### 1. Install Dependencies

```bash
cd web
pip install -r requirements.txt
```

### 2. Ensure Model is Trained

Make sure you have a trained model in `../models/`:
- `best_model_advanced.pth` (preferred)
- Or `best_model.pth` (basic model)

If you don't have a model, train one first:
```bash
cd ../src
python train_advanced.py
```

### 3. Run the Web App

```bash
python app.py
```

### 4. Open in Browser

Navigate to: **http://localhost:5000**

## Usage

1. **Upload Image**: Drag & drop or click to browse
2. **Analyze**: Click "Detect Shape" button
3. **View Results**: See prediction with confidence scores

## API Endpoints

### Health Check
```
GET /api/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "classes": ["circle", "rectangle", "square", "triangle"]
}
```

### Predict
```
POST /api/predict
```

Request:
- Method: `multipart/form-data`
- Field: `file` (image file)

Response:
```json
{
  "success": true,
  "prediction": "circle",
  "confidence": 95.6,
  "all_predictions": {
    "circle": 95.6,
    "square": 2.3,
    "triangle": 1.5,
    "rectangle": 0.6
  },
  "processing_time": 34.2
}
```

## Project Structure

```
web/
├── app.py                 # Flask backend
├── requirements.txt       # Python dependencies
├── static/
│   ├── css/
│   │   └── style.css     # Modern styling
│   ├── js/
│   │   └── main.js       # Frontend logic
│   └── uploads/          # Temporary uploads
└── templates/
    └── index.html        # Main page
```

## Configuration

Edit `app.py` to customize:

```python
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max file size
app.config['UPLOAD_FOLDER'] = 'static/uploads'        # Upload directory
```

## Deployment

### Heroku

```bash
# Create Procfile
echo "web: python app.py" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
```

### Render

1. Connect your GitHub repository
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `python app.py`

### Docker

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

## Troubleshooting

### Model not found
**Error**: "Model not loaded"
**Solution**: Train a model first or ensure model path is correct

### CUDA out of memory
**Solution**: Model will automatically fall back to CPU

### Port already in use
**Solution**: Change port in `app.py`:
```python
app.run(debug=True, port=5001)
```

## Browser Compatibility

- ✅ Chrome/Edge (recommended)
- ✅ Firefox
- ✅ Safari
- ✅ Mobile browsers

## Technologies Used

- **Backend**: Flask 3.0
- **ML**: PyTorch 2.0
- **Frontend**: Vanilla JavaScript
- **Styling**: Modern CSS3

## License

Same as main project license.

## Support

For issues, visit: https://github.com/RadwanSusan/AI-shape-detector-model/issues

---

**Built with ❤️ using PyTorch**
