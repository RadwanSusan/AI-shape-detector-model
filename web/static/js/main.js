// ============================================
// AI Shape Detector - Frontend JavaScript
// ============================================

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewArea = document.getElementById('previewArea');
const previewImage = document.getElementById('previewImage');
const analyzeBtn = document.getElementById('analyzeBtn');
const clearBtn = document.getElementById('clearBtn');
const loadingState = document.getElementById('loadingState');
const resultsSection = document.getElementById('resultsSection');

// Result elements
const shapeIcon = document.getElementById('shapeIcon');
const predictedShape = document.getElementById('predictedShape');
const confidenceBadge = document.getElementById('confidenceBadge');
const confidenceValue = document.getElementById('confidenceValue');
const predictionsList = document.getElementById('predictionsList');
const processingTime = document.getElementById('processingTime');

// State
let selectedFile = null;

// Shape icons mapping
const shapeIcons = {
    'circle': '⬤',
    'square': '◼',
    'triangle': '▲',
    'rectangle': '▭'
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
});

function setupEventListeners() {
    // Click to upload
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    // File selection
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);

    // Buttons
    analyzeBtn.addEventListener('click', analyzeImage);
    clearBtn.addEventListener('click', resetInterface);
}

// === File Handling ===

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file && isValidImageFile(file)) {
        selectedFile = file;
        displayPreview(file);
    } else {
        showError('Please select a valid image file (PNG, JPG, JPEG, GIF, BMP)');
    }
}

function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');

    const file = e.dataTransfer.files[0];
    if (file && isValidImageFile(file)) {
        selectedFile = file;
        fileInput.files = e.dataTransfer.files; // Update file input
        displayPreview(file);
    } else {
        showError('Please drop a valid image file');
    }
}

function isValidImageFile(file) {
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/bmp'];
    return validTypes.includes(file.type);
}

function displayPreview(file) {
    const reader = new FileReader();

    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadArea.style.display = 'none';
        previewArea.style.display = 'block';
        resultsSection.style.display = 'none';
    };

    reader.readAsDataURL(file);
}

// === Analysis ===

async function analyzeImage() {
    if (!selectedFile) {
        showError('No file selected');
        return;
    }

    // Show loading state
    previewArea.style.display = 'none';
    loadingState.style.display = 'block';
    resultsSection.style.display = 'none';

    // Prepare form data
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            displayResults(data);
        } else {
            showError(data.message || 'Prediction failed');
            resetToPreview();
        }

    } catch (error) {
        console.error('Error:', error);
        showError('Failed to analyze image. Please try again.');
        resetToPreview();
    }
}

function displayResults(data) {
    // Hide loading, show results
    loadingState.style.display = 'none';
    resultsSection.style.display = 'block';
    previewArea.style.display = 'block';

    // Update shape icon
    const shape = data.prediction.toLowerCase();
    shapeIcon.textContent = shapeIcons[shape] || '❓';

    // Update predicted shape
    predictedShape.textContent = data.prediction.toUpperCase();

    // Update confidence
    confidenceValue.textContent = data.confidence.toFixed(1);

    // Set confidence color
    const confidenceClass = getConfidenceClass(data.confidence);
    confidenceValue.className = `confidence-value ${confidenceClass}`;

    // Update processing time
    processingTime.textContent = data.processing_time.toFixed(0);

    // Display all predictions
    displayAllPredictions(data.all_predictions);

    // Scroll to results
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 100);
}

function displayAllPredictions(predictions) {
    predictionsList.innerHTML = '';

    // Sort predictions by confidence (already sorted from backend)
    Object.entries(predictions).forEach(([shape, confidence]) => {
        const item = document.createElement('div');
        item.className = 'prediction-item';

        const label = document.createElement('span');
        label.className = 'prediction-label';
        label.textContent = shape.charAt(0).toUpperCase() + shape.slice(1);

        const barContainer = document.createElement('div');
        barContainer.className = 'prediction-bar-container';

        const bar = document.createElement('div');
        bar.className = 'prediction-bar';
        bar.style.width = '0%'; // Start at 0 for animation

        const percentage = document.createElement('span');
        percentage.className = 'prediction-percentage';
        percentage.textContent = `${confidence.toFixed(1)}%`;

        bar.appendChild(percentage);
        barContainer.appendChild(bar);
        item.appendChild(label);
        item.appendChild(barContainer);
        predictionsList.appendChild(item);

        // Animate bar after a short delay
        setTimeout(() => {
            bar.style.width = `${confidence}%`;
        }, 50);
    });
}

function getConfidenceClass(confidence) {
    if (confidence >= 90) return 'confidence-high';
    if (confidence >= 70) return 'confidence-good';
    if (confidence >= 50) return 'confidence-medium';
    return 'confidence-low';
}

// === UI Controls ===

function resetInterface() {
    selectedFile = null;
    fileInput.value = '';
    uploadArea.style.display = 'block';
    previewArea.style.display = 'none';
    loadingState.style.display = 'none';
    resultsSection.style.display = 'none';
    previewImage.src = '';

    // Scroll to upload area
    uploadArea.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function resetToPreview() {
    loadingState.style.display = 'none';
    previewArea.style.display = 'block';
    resultsSection.style.display = 'none';
}

function showError(message) {
    alert(message);
    console.error('Error:', message);
}

// === Utility Functions ===

// Add smooth scrolling for the whole page
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({ behavior: 'smooth' });
        }
    });
});

// Prevent default drag behavior on the whole page
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    document.body.addEventListener(eventName, (e) => {
        if (e.target !== uploadArea && !uploadArea.contains(e.target)) {
            e.preventDefault();
            e.stopPropagation();
        }
    }, false);
});

console.log('🔷 AI Shape Detector - Ready!');
