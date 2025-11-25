# 🐟 Multiclass Fish Image Classification

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A deep learning project that classifies fish images into multiple species using CNN architectures and transfer learning. The project includes model training, comprehensive evaluation, and a user-friendly Streamlit web application for real-time predictions.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Models Implemented](#models-implemented)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training Models](#training-models)
  - [Running the Streamlit App](#running-the-streamlit-app)
- [Model Performance](#model-performance)
- [Technologies Used](#technologies-used)
- [Results and Visualizations](#results-and-visualizations)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## 🎯 Overview

This project addresses the challenge of automated fish species identification using computer vision and deep learning. The system:

- **Trains 6 different models** (1 custom CNN + 5 transfer learning models)
- **Evaluates and compares** their performance using multiple metrics
- **Deploys the best model** (MobileNet with 99.4% accuracy) in a Streamlit web app
- **Provides real-time predictions** with confidence scores

### Business Use Cases

1. **Marine Biology Research**: Automated species identification for biodiversity studies
2. **Fisheries Management**: Quick identification for sustainable fishing practices
3. **Aquarium Management**: Species verification and cataloging
4. **Educational Tools**: Interactive learning platform for ichthyology students

## ✨ Features

- **Multiple Model Architectures**: Compare CNN from scratch with pre-trained models
- **Comprehensive Data Augmentation**: Rotation, zoom, flipping for robust training
- **Detailed Evaluation Metrics**: Accuracy, precision, recall, F1-score, confusion matrix
- **Interactive Web App**: Upload images and get instant predictions
- **Model Comparison**: Visual comparison of all models' performance
- **Production-Ready**: Saved models in `.h5` and `.keras` formats

## 🧠 Models Implemented

| Model | Type | Test Accuracy | Parameters |
|-------|------|---------------|------------|
| **Custom CNN** | From Scratch | ~XX.X% | ~X.XM |
| **VGG16** | Transfer Learning | ~XX.X% | ~XXM |
| **ResNet50** | Transfer Learning | ~XX.X% | ~XXM |
| **MobileNet** | Transfer Learning | **99.4%** ⭐ | ~X.XM |
| **InceptionV3** | Transfer Learning | ~XX.X% | ~XXM |
| **EfficientNetB0** | Transfer Learning | ~XX.X% | ~XXM |

> **Best Model**: MobileNet achieved the highest accuracy and was selected for deployment.

## 📊 Dataset

The dataset consists of fish images organized into 11 species categories:

```
fish_data/
├── train/          # Training images
│   ├── species_1/
│   ├── species_2/
│   └── ...
├── val/            # Validation images
│   ├── species_1/
│   ├── species_2/
│   └── ...
└── test/           # Test images
    ├── species_1/
    ├── species_2/
    └── ...
```

**Data Preprocessing**:
- Images resized to 224×224 pixels
- Normalized to [0, 1] range
- Augmentation: rotation (30°), zoom (0.2), horizontal/vertical flip

## 📁 Project Structure

```
multiclass-fish-classification/
│
├── MFC_test.ipynb              # Complete training and evaluation notebook
├── MFC_App.py                  # Streamlit web application
├── class_labels.json           # Fish species labels
│
├── Custom CNN/
│   ├── custom_cnn_model.h5
│   └── custom_cnn_model.keras
│
├── VGG16/
│   ├── vgg_finetuned_model.h5
│   └── vgg_finetuned_model.keras
│
├── ResNet50/
│   ├── resnet_finetuned_model.h5
│   └── resnet_finetuned_model.keras
│
├── MobileNet/                  # ⭐ Best Model
│   ├── mobilenet_finetuned_model.h5
│   └── mobilenet_finetuned_model.keras
│
├── InceptionV3/
│   ├── inception_finetuned_model.h5
│   └── inception_finetuned_model.keras
│
├── EfficientNetB0/
│   ├── efficientnet_finetuned_model.h5
│   └── efficientnet_finetuned_model.keras
│
├── requirements.txt
└── README.md
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- GPU (optional, but recommended for training)

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/multiclass-fish-classification.git
cd multiclass-fish-classification
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Requirements

Create a `requirements.txt` file with:

```
tensorflow>=2.10.0
streamlit>=1.25.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
pillow>=9.3.0
scikit-learn>=1.2.0
```

## 💻 Usage

### Training Models

1. **Prepare your dataset**: Organize images into `train/`, `val/`, and `test/` directories as shown in the [Dataset](#dataset) section.

2. **Update data paths** in the notebook:
```python
train_dir = "path/to/your/fish_data/train"
val_dir = "path/to/your/fish_data/val"
test_dir = "path/to/your/fish_data/test"
```

3. **Run the training notebook**:
   - Open `MFC_test.ipynb` in Jupyter Notebook or JupyterLab
   - Run all cells to train all 6 models
   - Models will be saved automatically in their respective folders

4. **Training parameters**:
   - Image size: 224×224
   - Batch size: 32
   - Custom CNN epochs: 20
   - Transfer learning epochs: 5 (with fine-tuning)

### Running the Streamlit App

1. **Ensure the MobileNet model is available**:
```bash
# Check if the model file exists
ls MobileNet/mobilenet_finetuned_model.h5
```

2. **Launch the application**:
```bash
streamlit run MFC_App.py
```

3. **Use the app**:
   - Open your browser at `http://localhost:8501`
   - Upload a fish image (JPG/PNG)
   - View the prediction and confidence scores
   - Explore top-3 predictions and raw probabilities

## 📈 Model Performance

### Evaluation Metrics

All models were evaluated on the test set using:

- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation of classification performance

### Training Strategy

**Custom CNN**:
- Built from scratch with 3 convolutional blocks
- BatchNormalization for stable training
- Dropout (0.5) to prevent overfitting
- 20 epochs with Adam optimizer (lr=0.0001)

**Transfer Learning Models**:
- Initialized with ImageNet pre-trained weights
- Fine-tuned last layers (model-specific)
- GlobalAveragePooling2D for spatial dimension reduction
- Dense layers with Dropout (0.5)
- 5 epochs with Adam optimizer (lr=1e-5)

### Model Selection Criteria

MobileNet was selected as the deployment model because:
1. ✅ **Highest accuracy**: 99.4% on test set
2. ✅ **Lightweight**: Small model size (~4MB)
3. ✅ **Fast inference**: Quick predictions for real-time use
4. ✅ **Balanced performance**: High precision and recall across all classes

## 🛠️ Technologies Used

- **Deep Learning**: TensorFlow, Keras
- **Web Framework**: Streamlit
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib
- **Model Evaluation**: scikit-learn
- **Image Processing**: PIL (Pillow)

## 📊 Results and Visualizations

The training notebook generates:

1. **Accuracy/Loss Plots**: Training and validation curves for each model
2. **Confusion Matrices**: Per-model classification performance
3. **Classification Reports**: Precision, recall, F1-score for each species
4. **Model Comparison Table**: Side-by-side metrics comparison

Example visualizations are saved during training and can be reproduced by running the notebook.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: [Mention dataset source if applicable]
- **Pre-trained Models**: TensorFlow/Keras Applications
- **Inspiration**: Marine biology research and sustainable fishing practices

## 📧 Contact

**Your Name** - [your.email@example.com](mailto:your.email@example.com)

Project Link: [https://github.com/yourusername/multiclass-fish-classification](https://github.com/yourusername/multiclass-fish-classification)

---

## 🎓 Skills Demonstrated

- Deep Learning and Neural Networks
- Transfer Learning and Fine-tuning
- Data Augmentation Techniques
- Model Evaluation and Comparison
- Web Application Deployment
- Python Programming (PEP 8 compliant)
- Git Version Control

---

### 📌 Quick Start Commands

```bash
# Clone and setup
git clone https://github.com/yourusername/multiclass-fish-classification.git
cd multiclass-fish-classification
pip install -r requirements.txt

# Run the app
streamlit run MFC_App.py
```

---

**⭐ If you find this project helpful, please give it a star!**
