# Satellite Image Classification with PyTorch

A comprehensive deep learning solution for classifying satellite images using PyTorch. This project implements both transfer learning with ResNet-50 and a custom CNN architecture, providing a complete framework for satellite image analysis.

## 🎯 Project Overview

This project tackles multi-class satellite image classification using state-of-the-art deep learning techniques. It's designed to be both educational and production-ready, featuring:

- **Transfer Learning**: Pre-trained ResNet-50 for quick, high-accuracy results
- **Custom CNN**: Hand-crafted architecture for learning and experimentation
- **Complete Pipeline**: Data loading, preprocessing, training, and evaluation
- **Comprehensive Evaluation**: Accuracy metrics, confusion matrices, and training visualizations

## 🚀 Features

- ✅ **Dual Model Architecture**: ResNet-50 (transfer learning) + Custom CNN
- ✅ **Advanced Data Augmentation**: Rotation, flipping, color jittering
- ✅ **GPU Acceleration**: Automatic CUDA detection and utilization
- ✅ **Training Monitoring**: Real-time loss and accuracy tracking
- ✅ **Model Persistence**: Save and load trained models
- ✅ **Detailed Evaluation**: Classification reports and confusion matrices
- ✅ **Visualization**: Training curves and performance plots

## 📋 Requirements

### Dependencies
```bash
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
Pillow>=8.3.0
tqdm>=4.62.0
```

### System Requirements
- **Python**: 3.7 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 2GB for models and data
- **GPU**: CUDA-compatible GPU recommended (optional but faster)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd satellite-image-classification
   ```

2. **Create virtual environment**
   ```bash
   python -m venv satellite_env
   source satellite_env/bin/activate  # On Windows: satellite_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install torch torchvision matplotlib seaborn scikit-learn pillow tqdm numpy
   ```

## 📁 Data Structure

Organize your satellite image data in the following directory structure:

```
satellite_data/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class3/
│   └── ...
└── classN/
    └── ...
```

**Supported formats**: `.jpg`, `.jpeg`, `.png`

## 🏃‍♂️ Quick Start

1. **Prepare your data** according to the structure above

2. **Update the data path** in the main script:
   ```python
   DATA_DIR = "path/to/your/satellite/data"
   ```

3. **Run the training script**:
   ```bash
   python satellite_classifier.py
   ```

4. **Monitor training progress** - the script will display:
   - Real-time training/validation loss and accuracy
   - Progress bars for each epoch
   - Model comparison results
   - Training visualization plots

## 🔧 Configuration

### Key Parameters

```python
# Model Configuration
BATCH_SIZE = 32          # Adjust based on GPU memory
NUM_EPOCHS = 20          # Training epochs
LEARNING_RATE = 0.001    # Learning rate

# Data Split
TRAIN_SPLIT = 0.6        # 60% for training
VAL_SPLIT = 0.2          # 20% for validation  
TEST_SPLIT = 0.2         # 20% for testing
```

### Model Architectures

#### ResNet-50 (Transfer Learning)
- **Base**: Pre-trained ResNet-50 on ImageNet
- **Custom Head**: 2048 → 512 → num_classes
- **Regularization**: Dropout (0.5, 0.3)
- **Best for**: Quick results, limited data

#### Custom CNN
- **Architecture**: Progressive depth increase (64→128→256→512)
- **Features**: Batch normalization, ReLU activation, MaxPooling
- **Regularization**: Dropout (0.5, 0.3)
- **Best for**: Learning, experimentation, custom requirements

## 📊 Model Performance

### Expected Results
- **ResNet-50**: 85-95% accuracy (typical for satellite data)
- **Custom CNN**: 75-90% accuracy (depends on data complexity)
- **Training Time**: 
  - GPU: ~10-30 minutes (depends on dataset size)
  - CPU: ~2-6 hours

### Evaluation Metrics
- **Accuracy**: Overall classification accuracy
- **Precision/Recall**: Per-class performance metrics
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown

## 📈 Outputs

The script generates several outputs:

### Saved Models
- `resnet_satellite_classifier.pth` - ResNet model weights
- `custom_cnn_satellite_classifier.pth` - Custom CNN weights

### Visualizations
- **Training Curves**: Loss and accuracy over epochs
- **Confusion Matrix**: Classification performance heatmap
- **Model Comparison**: Side-by-side performance analysis

### Console Outputs
```
Epoch [1/20]
Train Loss: 1.2345, Train Acc: 67.50%
Val Loss: 1.1234, Val Acc: 72.30%
------------------------------------------------------------
ResNet Accuracy: 0.8945
Custom CNN Accuracy: 0.8234
```

## 🎛️ Advanced Configuration

### Data Augmentation
Customize augmentation in the transform pipeline:
```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),      # Horizontal flip probability
    transforms.RandomVerticalFlip(p=0.5),       # Vertical flip probability  
    transforms.RandomRotation(degrees=15),       # Rotation range
    transforms.ColorJitter(brightness=0.2),     # Color augmentation
    # Add more augmentations as needed
])
```

### Learning Rate Scheduling
```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
# Reduces learning rate by factor of 0.1 every 7 epochs
```

### Model Customization
Modify the custom CNN architecture:
```python
# Add more layers
nn.Conv2d(512, 1024, kernel_size=3, padding=1),
nn.BatchNorm2d(1024),
nn.ReLU(inplace=True),

# Change activation functions
nn.LeakyReLU(0.2, inplace=True)  # Instead of ReLU

# Adjust dropout rates
nn.Dropout(0.7)  # Higher dropout for more regularization
```

## 🔍 Troubleshooting

### Common Issues

**Out of Memory Error**
```python
# Reduce batch size
BATCH_SIZE = 16  # or 8

# Enable gradient checkpointing
torch.cuda.empty_cache()
```

**Low Accuracy**
- Check data quality and labels
- Increase number of epochs
- Adjust learning rate (try 0.0001)
- Add more data augmentation

**Slow Training**
- Use GPU if available
- Increase batch size (if memory allows)
- Reduce image resolution temporarily

**Overfitting**
- Increase dropout rates
- Add more data augmentation
- Reduce model complexity
- Use early stopping

## 📚 Understanding the Code

### Key Components

1. **SatelliteImageDataset**: Custom PyTorch Dataset class
2. **ResNetClassifier**: Transfer learning implementation
3. **CustomCNN**: From-scratch CNN architecture
4. **train_model()**: Complete training loop with validation
5. **evaluate_model()**: Comprehensive model evaluation

### Training Pipeline
```
Data Loading → Preprocessing → Model Creation → Training Loop → Evaluation → Visualization
```

### Data Flow
```
Images (PIL) → Transforms → Tensors → Model → Predictions → Loss → Gradients → Weight Updates
```
