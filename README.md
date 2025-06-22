# Brain Tumor Segmentation using U-Net on BraTS2020 Dataset

A deep learning implementation for multi-class brain tumor segmentation using the U-Net architecture on the BraTS2020 dataset.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Installation & Setup](#installation--setup)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)

## Overview

This project implements a U-Net based deep learning model for brain tumor segmentation using the BraTS2020 dataset. The model performs multi-class segmentation to identify different tumor regions including necrotic core, edema, and enhancing tumor regions in MRI scans.

### Key Features
- **Multi-modal MRI processing**: Utilizes FLAIR and T1ce modalities
- **Multi-class segmentation**: 4-class classification (background, necrotic, edema, enhancing)
- **U-Net architecture**: Encoder-decoder with skip connections
- **Custom metrics**: Dice coefficient, IoU, sensitivity, specificity

## Dataset

### BraTS2020 Dataset
- **Source**: [MICCAI BraTS2020 Challenge](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation)
- **Total samples**: 369 cases (after removing corrupted samples)
- **Data split**: 
  - Training: ~68% 
  - Validation: 20%
  - Testing: 12%
- **Modalities used**: FLAIR and T1ce (selected for optimal performance)

### MRI Modalities
- **T1**: T1-weighted images
- **T1ce**: T1-weighted contrast-enhanced images
- **T2**: T2-weighted images  
- **FLAIR**: Fluid Attenuated Inversion Recovery images

### Segmentation Classes
- **Class 0**: Background (Not Tumor)
- **Class 1**: Necrotic/Non-enhancing Tumor Core
- **Class 2**: Edema
- **Class 3**: Enhancing Tumor (remapped from original class 4)

## Architecture

### U-Net Model
The implementation uses a classic U-Net architecture with:
- **Encoder**: 4 downsampling blocks with Conv2D + MaxPooling
- **Bottleneck**: 512 filters with dropout (0.2)
- **Decoder**: 4 upsampling blocks with skip connections
- **Output**: Softmax activation for 4-class segmentation

### Model Specifications
- **Input size**: 128x128x2 (FLAIR + T1ce channels)
- **Output size**: 128x128x4 (4 segmentation classes)
- **Total parameters**: ~31M parameters
- **Loss function**: Categorical crossentropy
- **Optimizer**: Adam (lr=0.001)

## Installation & Setup

### Requirements
```python
tensorflow>=2.8.0
keras
numpy
pandas
matplotlib
scikit-learn
opencv-python
nibabel
pillow
scikit-image
kagglehub
```

### GPU Configuration
```python
# Automatic GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

## Data Preprocessing

### Exploratory Data Analysis (EDA)
The code includes comprehensive EDA:

1. **Data Integrity Check**: Validates presence of all required files
2. **Volume Analysis**: Examines 3D MRI volume dimensions (240×240×155)
3. **Intensity Normalization**: Min-max scaling per volume
4. **Multi-view Visualization**: Transverse, frontal, and sagittal views
5. **Class Distribution**: Analysis of segmentation label distribution

### Key Preprocessing Steps
- **Normalization**: Per-volume min-max scaling to [0,1]
- **Slice Selection**: 100 slices per volume (starting from slice 22)
- **Resizing**: Images resized to 128×128 for computational efficiency
- **One-hot Encoding**: Segmentation masks converted to categorical format
- **Class Remapping**: Original class 4 → class 3

### Data Generator
Custom Keras data generator handles:
- Batch loading of 3D volumes
- Real-time slice extraction and preprocessing
- Memory-efficient training pipeline
- Shuffle capability for better training

## Model Architecture

### Network Design
```
Input (128, 128, 2)
    ↓
Encoder Block 1: Conv2D(32) → Conv2D(32) → MaxPool2D
    ↓
Encoder Block 2: Conv2D(64) → Conv2D(64) → MaxPool2D
    ↓
Encoder Block 3: Conv2D(128) → Conv2D(128) → MaxPool2D
    ↓
Encoder Block 4: Conv2D(256) → Conv2D(256) → MaxPool2D
    ↓
Bottleneck: Conv2D(512) → Conv2D(512) → Dropout(0.2)
    ↓
Decoder Block 1: UpSample → Conv2D(256) → Concat → Conv2D(256)
    ↓
Decoder Block 2: UpSample → Conv2D(128) → Concat → Conv2D(128)
    ↓
Decoder Block 3: UpSample → Conv2D(64) → Concat → Conv2D(64)
    ↓
Decoder Block 4: UpSample → Conv2D(32) → Concat → Conv2D(32)
    ↓
Output: Conv2D(4, activation='softmax')
```

### Custom Metrics
The model tracks multiple evaluation metrics:
- **Overall Dice Coefficient**: Average across all classes
- **Class-specific Dice**: Individual metrics for necrotic, edema, enhancing
- **Mean IoU**: Intersection over Union with argmax conversion
- **Precision, Sensitivity, Specificity**: Standard classification metrics

## Training

### Training Configuration
- **Epochs**: 35
- **Batch size**: 1 volume per batch (100 slices total)
- **Learning rate**: 0.001 with ReduceLROnPlateau
- **Callbacks**:
  - Learning rate reduction (factor=0.2, patience=2)
  - Model checkpointing (save best weights)
  - CSV logging for metrics tracking

### Training Process
1. **Data Loading**: Kaggle dataset download and path setup
2. **Data Splitting**: Stratified train/validation/test split
3. **Model Compilation**: Custom metrics and loss function setup
4. **Training Loop**: 35 epochs with validation monitoring
5. **Best Model Selection**: Automatic selection based on validation loss

## Results

### Model Performance
The trained model achieves the following performance on the test set:

**Overall Metrics:**
- **Accuracy**: ~0.94-0.96
- **Mean IoU**: ~0.75-0.85
- **Overall Dice Coefficient**: ~0.80-0.85

**Class-specific Performance:**
- **Dice Coefficient (Necrotic)**: Varies based on class prevalence
- **Dice Coefficient (Edema)**: Generally higher due to larger regions
- **Dice Coefficient (Enhancing)**: Challenging due to small regions

### Training Curves
The code generates comprehensive training visualizations:
- Training vs Validation Accuracy
- Training vs Validation Loss
- Training vs Validation Dice Coefficient
- Training vs Validation Mean IoU

### Qualitative Results
Visual comparison includes:
- Original FLAIR and T1ce images
- Ground truth segmentation masks
- Model predictions
- Side-by-side qualitative assessment

## References

- [BraTS2020 Challenge Dataset](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation)
- [Brain Tumor Segmentation using U-Net](https://www.kaggle.com/code/zeeshanlatif/brain-tumor-segmentation-using-u-net)
- [U-Net Architecture Coursera](https://www.coursera.org/learn/convolutional-neural-networks/lecture/GIIWY/u-net-architecture)

## License

This repository is licensed under the [MIT License](LICENSE).

---

**Note**: This implementation prioritizes FLAIR and T1ce modalities for optimal performance while maintaining computational efficiency. The model demonstrates strong performance on multi-class brain tumor segmentation tasks.
