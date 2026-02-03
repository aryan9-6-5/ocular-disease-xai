# Ocular Disease Recognition: Multi-Method Approach

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow.svg)

> A comprehensive machine learning and deep learning project for automated detection of multiple ocular diseases from fundus images, featuring both traditional computer vision (LBP) and state-of-the-art deep learning approaches with explainability.

---

##  Table of Contents
- [Overview](#overview)
- [Project Approaches](#project-approaches)
- [Features](#features)
- [Results Summary](#results-summary)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Usage](#usage)
- [Model Comparison](#model-comparison)
- [Explainability](#explainability)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

---

##  Overview

Ocular diseases, including diabetic retinopathy, glaucoma, cataracts, and age-related macular degeneration (AMD), represent significant causes of preventable blindness worldwide. This project develops automated classification systems to detect multiple ocular diseases from retinal fundus images using both traditional computer vision and advanced deep learning techniques.

**Key Objectives:**
- Detect and classify 8 ocular conditions: Normal, Diabetic Retinopathy, Glaucoma, Cataract, AMD, Hypertension, Myopia, and Others
- Compare traditional feature extraction (LBP) with deep learning approaches
- Provide explainable AI (XAI) insights for clinical decision support
- Achieve clinically relevant performance metrics for diagnostic screening
- Create accessible tools for early detection and prevention

---

##  Project Approaches

This project implements **two complementary methodologies**:

### 1. **Traditional Computer Vision Approach**
- **Method**: Local Binary Patterns (LBP) for texture feature extraction
- **Classification**: SVM, Random Forest, MLP classifiers
- **Scope**: Binary classification (Cataract vs. Normal)
- **Status**:  In Development

### 2. **Deep Learning + Explainable AI Approach**
- **Method**: Transfer learning with ResNet50, VGG19, Vision Transformer
- **Classification**: Multi-label classification (8 disease classes)
- **Explainability**: Grad-CAM and SHAP for visual interpretability
- **Status**:  Completed (ResNet50 model)

---

##  Features

### General Capabilities
-  Multi-label classification (8 disease classes)
-  Binary classification (cataract detection)
-  State-of-the-art deep learning models
-  Traditional feature extraction with LBP
-  Comprehensive data augmentation pipeline
-  GPU-accelerated training (PyTorch)

### Explainability & Visualization
-  **Grad-CAM**: Visual attention heatmaps highlighting pathological regions
-  **SHAP**: Pixel-level attribution for class predictions
-  Comparative model analysis
-  Interactive Streamlit interface (coming soon)

### Data Processing
-  CLAHE (Contrast Limited Adaptive Histogram Equalization)
-  Rotation, flip, brightness/contrast augmentation
-  Gaussian blur and noise injection
-  Class balancing via stratified sampling
-  Multi-scale LBP extraction

---

##  Results Summary

### Deep Learning Models (Multi-Class Classification)

**ResNet50 Performance on Test Set (10 epochs):**

| Disease        | Precision | Recall | F1-Score | AUC   |
|----------------|-----------|--------|----------|-------|
| Normal         | 0.875     | 0.919  | 0.896    | 0.977 |
| Diabetes       | 0.919     | 0.879  | 0.898    | 0.971 |
| Glaucoma       | 0.941     | 0.898  | 0.919    | 0.987 |
| Cataract       | 0.942     | 0.922  | 0.932    | 0.987 |
| AMD            | 0.962     | 0.848  | 0.901    | 0.983 |
| Hypertension   | 0.912     | 0.806  | 0.856    | 0.965 |
| Myopia         | 0.988     | 0.945  | 0.966    | 0.993 |
| Others         | 0.927     | 0.836  | 0.879    | 0.966 |
| **Macro Avg**  | **0.933** | **0.882** | **0.906** | **0.979** |

### Binary Classification Models (Cataract Detection)

| Model | Training Accuracy | Training Loss | Validation Accuracy | Validation Loss |
|-------|------------------|---------------|---------------------|-----------------|
| **ResNet50** | 0.9710 | 0.0460 | **1.0000** | **0.0082** |
| **VGG19** | 1.0000 | 0.0049 | 0.9444 | 0.0996 |
| **Vision Transformer** | 0.7581 | 0.7465 | 0.8571 | 0.3453 |

** Best Model: ResNet50**
-  Perfect validation accuracy (100%) for binary cataract detection
-  Macro F1 Score of 0.906 for multi-class classification
-  AUC of 0.979 across all disease classes
-  Excellent generalization with minimal overfitting

---

##  Dataset

**ODIR-5K (Ocular Disease Intelligent Recognition)**

- **Source**: Peking University / Kaggle Competition Dataset
- **Size**: 5,000 patients with ~10,000 fundus images (left and right eyes)
- **Format**: Color fundus photographs (various resolutions)
- **Annotations**: Multi-label annotations for 8 conditions
- **Classes**: Normal, Diabetic Retinopathy, Glaucoma, Cataract, AMD, Hypertension, Myopia, Others

**Download Links:**
- Images: [Kaggle ODIR-5K Dataset](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k)
- Annotations: ODIR-5K_Training_Annotations(Updated)_V2.xlsx

**Dataset Characteristics:**
- Multi-label dataset (patients may have multiple diseases)
- Balanced training via stratified sampling
- Train-validation-test split: 70-15-15

---

##  Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended for deep learning)
- pip package manager
- 10GB+ free disk space for dataset

### Setup Instructions

```bash
# Clone the repository
git clone https://github.com/yourusername/ocular-disease-recognition.git
cd ocular-disease-recognition

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### Requirements

```txt
# Core Dependencies
numpy>=1.21.0
pandas>=1.3.0
opencv-python>=4.5.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0

# Deep Learning
torch>=2.0.0
torchvision>=0.15.0
tensorflow>=2.8.0  # Optional, for alternative implementations

# Explainability
shap>=0.41.0
captum>=0.6.0  # For Grad-CAM

# Image Processing
Pillow>=8.3.0
scikit-image>=0.18.0
albumentations>=1.3.0

# Utilities
tqdm>=4.62.0
jupyter>=1.0.0
ipykernel>=6.0.0
openpyxl>=3.0.0  # For Excel annotations

# Optional - Web Interface
streamlit>=1.20.0  # Coming soon
```

---

## Project Structure

```text
ocular-disease-xai/
├── data/
│   ├── raw/
│   │   └── ODIR-5K/
│   │       └── Training_Images/
│   └── processed/
├── src/
│   ├── data_preprocessing.py
│   ├── models/
│   └── explainability/
│       ├── gradcam.py
│       └── shap_utils.py
├── notebooks/
│   └── model.ipynb
├── models/
│   └── ocular_resnet50_final_10epochs.pth
├── results/
│   └── metrics/
├── app/
├── requirements.txt
└── README.md
```

---

## 🔧 Methodology

### Approach 1: Deep Learning + Explainable AI

#### **Architecture: ResNet50 with Transfer Learning**

```
Input (256×256×3) 
    ↓
ResNet50 Backbone (ImageNet pretrained)
    ↓
Global Average Pooling
    ↓
Dense Layer (128 units, ReLU)
    ↓
Dropout (0.5)
    ↓
Output Layer (8 units, Sigmoid) → Multi-label predictions
```

#### **Training Configuration:**
- **Backbone**: ResNet50 pretrained on ImageNet
- **Optimizer**: Adam (learning rate = 1e-4)
- **Loss Function**: BCEWithLogitsLoss (for multi-label)
- **Batch Size**: 32
- **Epochs**: 10 (extendable to 50 with early stopping)
- **Data Augmentation**:
  - CLAHE preprocessing
  - Horizontal flip (p=0.5)
  - Rotation (±15°)
  - Brightness/contrast adjustment
  - Gaussian blur

#### **Explainability Methods:**

1. **Grad-CAM (Gradient-weighted Class Activation Mapping)**
   - Highlights image regions influencing predictions
   - Identifies pathological features (e.g., optic disc for glaucoma, lens opacity for cataract)
   - Layer: Last convolutional layer of ResNet50

2. **SHAP (SHapley Additive exPlanations)**
   - Pixel-level contribution to each class prediction
   - GradientExplainer with background dataset
   - Generates attribution heatmaps

---

### Approach 2: Traditional Computer Vision (LBP)

#### **Local Binary Patterns (LBP) Feature Extraction**

**Algorithm:**
1. Convert fundus image to grayscale
2. For each pixel, compare with surrounding neighbors (8, 16, or 24 points)
3. Create binary pattern based on intensity comparison
4. Convert binary pattern to decimal (0-255)
5. Compute histogram of LBP codes as feature vector

**Multi-scale LBP Parameters:**
- **Radii**: 1, 2, 3 pixels
- **Points**: 8, 16, 24 neighbors
- **Method**: Uniform patterns (rotation-invariant)
- **Feature Dimension**: 59 bins × 3 scales = 177 features

#### **Classification Pipeline:**

```
Fundus Image 
    ↓
Preprocessing (Resize, Grayscale, CLAHE)
    ↓
Multi-scale LBP Extraction
    ↓
Feature Vector (177-dim)
    ↓
Classifier (SVM/RF/MLP)
    ↓
Prediction (Cataract vs. Normal)
```

**Classifier Options:**
- Support Vector Machine (RBF kernel, C=1.0, gamma='scale')
- Random Forest (100 estimators, max_depth=None)
- Multi-layer Perceptron (hidden layers: [128, 64])

---

##  Usage

### 1. Deep Learning Model Training

```python
# Train ResNet50 for multi-class classification
from src.training.train_dl import train_model
from src.models.resnet_model import OcularResNet50

# Initialize model
model = OcularResNet50(num_classes=8, pretrained=True)

# Train
train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=10,
    learning_rate=1e-4,
    save_path='models/ocular_resnet50.pth'
)
```

### 2. Explainability Visualization

```python
# Generate Grad-CAM heatmap
from src.explainability.gradcam import generate_gradcam

heatmap = generate_gradcam(
    model=model,
    img_tensor=img_tensor,
    target_class=3,  # Cataract
    layer_name='layer4'
)

# Generate SHAP attribution
from src.explainability.shap_utils import generate_shap

shap_values = generate_shap(
    model=model,
    img_tensor=img_tensor,
    background_batch=background_data
)
```

### 3. LBP Feature Extraction and Classification

```python
# Extract LBP features
from src.feature_extraction import extract_lbp_features
from sklearn.svm import SVC

# Load and preprocess image
features = extract_lbp_features(
    image_path='data/fundus_image.jpg',
    radii=[1, 2, 3],
    points=[8, 16, 24]
)

# Train SVM classifier
classifier = SVC(kernel='rbf', C=1.0, gamma='scale')
classifier.fit(X_train_lbp, y_train)

# Predict
prediction = classifier.predict([features])
```

### 4. Inference on New Images

```python
# Load trained ResNet50 model
import torch
from src.models.resnet_model import OcularResNet50

model = OcularResNet50(num_classes=8)
model.load_state_dict(torch.load('models/ocular_resnet50_final_10epochs.pth'))
model.eval()

# Predict
from src.utils import predict_image

results = predict_image(
    model=model,
    image_path='path/to/fundus_image.jpg',
    threshold=0.5
)

print(f"Predicted diseases: {results['diseases']}")
print(f"Probabilities: {results['probabilities']}")
```

### 5. Running Jupyter Notebooks

```bash
# Start Jupyter server
jupyter notebook

# Open notebooks:
# - model.ipynb: Deep learning experiments
# - Ocular_LBP.ipynb: LBP experiments
```

---

##  Model Comparison

### Performance Analysis

![Model Comparison](results/visualizations/comparison.png)

### Key Findings

#### **Multi-Class Classification (ResNet50)**
-  **Best Overall Performance**: Macro F1 = 0.906, AUC = 0.979
-  **Strongest Classes**: Myopia (F1=0.966), Cataract (F1=0.932), Glaucoma (F1=0.919)
-  **Challenging Classes**: Hypertension (F1=0.856) - requires more data

#### **Binary Classification (Cataract Detection)**

| Metric | ResNet50 | VGG19 | Vision Transformer |
|--------|----------|-------|-------------------|
| **Validation Accuracy** |  **100%** | 94.44% | 85.71% |
| **Validation Loss** |  **0.0082** | 0.0996 | 0.3453 |
| **Training Accuracy** | 97.10% |  **100%** | 75.81% |
| **Generalization** |  Excellent |  Slight overfitting |  Underfitting |

#### **LBP Approach (In Development)**
- Status: Feature extraction complete, classifier training pending
- Expected Performance: F1 ~0.75-0.85 for binary classification
- Advantages: Lightweight, interpretable, no GPU required

### Clinical Implications

**ResNet50 Deployment Recommendations:**
1. **Low false positive rate** → Minimal unnecessary referrals
2. **Robust across all diseases** → Comprehensive screening tool
3. **Explainable predictions** → Grad-CAM shows pathological regions

---

##  Explainability

### Grad-CAM Visualizations

Grad-CAM highlights regions of interest for each disease class:

- **Glaucoma**: Focus on optic disc and cup-to-disc ratio
- **Cataract**: Attention on lens opacity and clouding
- **Diabetic Retinopathy**: Microaneurysms, hemorrhages, exudates
- **AMD**: Macula and drusen deposits

**Example Usage:**
```python
from src.explainability.gradcam import visualize_gradcam

visualize_gradcam(
    model=model,
    image_path='data/test/glaucoma_case.jpg',
    class_idx=2, 
    save_path='results/visualizations/gradcam/glaucoma_example.png'
)
```

### SHAP Attribution Maps

SHAP provides pixel-level explanations:
- Positive contributions (red): Pixels supporting disease prediction
- Negative contributions (blue): Pixels opposing disease prediction
- Magnitude: Importance of each pixel

**Example Usage:**
```python
from src.explainability.shap_utils import explain_prediction

explanation = explain_prediction(
    model=model,
    image_path='data/test/dr_case.jpg',
    background_size=50
)
```

---

## 🔮 Future Work

### Short-term Goals
- [ ] Complete LBP classifier training and validation
- [ ] Implement k-fold cross-validation for all models
- [ ] Add confusion matrix visualizations
- [ ] Deploy Streamlit web interface for demos
- [ ] Hyperparameter tuning with Optuna/Ray Tune

### Medium-term Goals 
- [ ] Ensemble methods combining ResNet50 + VGG19
- [ ] Attention mechanisms for improved localization
- [ ] Extended training to 50 epochs with learning rate scheduling
- [ ] Multi-modal fusion (left + right eye images)
- [ ] Real-world validation with clinical partners

### Long-term Vision 
- [ ] Production API with FastAPI/Docker
- [ ] Mobile app for point-of-care screening
- [ ] Integration with electronic health records (EHR)
- [ ] Multi-center validation study
- [ ] FDA/CE regulatory pathway exploration
- [ ] Publication in medical imaging journal

---

**Dataset License:**  
The ODIR-5K dataset is subject to Kaggle competition terms and the original data providers' conditions. Please refer to the dataset documentation for usage restrictions.

**Citation:**
If you use this code in your research, please cite:
```bibtex
@software{ocular_disease_recognition_2024,
  author = {Your Name},
  title = {Ocular Disease Recognition: Multi-Method Approach},
  year = {2024},
  url = {https://github.com/yourusername/ocular-disease-recognition}
}
```

---

##  References

### Academic Papers

**Deep Learning for Medical Imaging:**
1. He, K., et al. (2016). "Deep Residual Learning for Image Recognition." *CVPR*.
2. Simonyan, K., & Zisserman, A. (2015). "Very Deep Convolutional Networks for Large-Scale Image Recognition." *ICLR*.
3. Dosovitskiy, A., et al. (2021). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." *ICLR*.

**Explainable AI:**
4. Selvaraju, R. R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." *ICCV*.
5. Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions." *NeurIPS*.

**Texture Analysis:**
6. Ojala, T., Pietikäinen, M., & Harwood, D. (1996). "A comparative study of texture measures with classification based on featured distributions." *Pattern Recognition*, 29(1), 51-59.

**Medical Applications:**
7. Gulshan, V., et al. (2016). "Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy." *JAMA*, 316(22), 2402-2410.
8. Li, L., et al. (2019). "Attention-based deep neural network for automatic detection of diabetic retinopathy." *Medical Image Analysis*, 53, 72-84.

### Datasets
- **ODIR-5K**: Peking University International Competition on Ocular Disease Intelligent Recognition (2019)
- Kaggle: https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k

### Tools and Libraries
- **PyTorch**: https://pytorch.org/
- **scikit-learn**: https://scikit-learn.org/
- **OpenCV**: https://opencv.org/
- **SHAP**: https://shap.readthedocs.io/
- **Captum**: https://captum.ai/

---

**Acknowledgments:**
- ODIR-5K dataset creators and Kaggle community
- Open-source contributors to PyTorch, scikit-learn, and related libraries
- Medical imaging research community

---