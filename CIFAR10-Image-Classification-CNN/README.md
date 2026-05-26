<div align="center">

# 🖼️ CIFAR-10 Image Classification using Custom CNN

A **VGG-inspired Convolutional Neural Network** built from scratch with TensorFlow/Keras to classify images across 10 categories — achieving **~83% test accuracy**.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

</div>

---

## 📌 Overview

This project implements a **custom CNN architecture** for multi-class image classification on the popular **CIFAR-10** benchmark dataset. The model is designed with regularization and training strategies to reduce overfitting and achieve strong generalization.

### What makes this project stand out?
- Trained **entirely from scratch** (no pretrained weights)
- Employs **multiple regularization techniques** in combination
- Full **evaluation pipeline** — confusion matrix, classification report, ROC curves
- Clean, modular, and well-commented code

---

## 📂 Dataset — CIFAR-10

- **60,000** color images of size **32×32** across **10 classes**
- **50,000** training | **10,000** testing
- Auto-downloaded via `tensorflow.keras.datasets`

| Label | Class | Label | Class |
|-------|------------|-------|------------|
| 0 | ✈️ Airplane | 5 | 🐶 Dog |
| 1 | 🚗 Automobile | 6 | 🐸 Frog |
| 2 | 🐦 Bird | 7 | 🐴 Horse |
| 3 | 🐱 Cat | 8 | 🚢 Ship |
| 4 | 🦌 Deer | 9 | 🚛 Truck |

---

## 🏗️ Model Architecture

> VGG-inspired design with progressive filter depth and global average pooling

```
Input (32×32×3)
       │
       ▼
┌─────────────────────────────────┐
│  Conv Block 1                   │
│  Conv2D(32) → BatchNorm         │
│  MaxPooling(2×2)                │
└─────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  Conv Block 2                   │
│  Conv2D(64) → BatchNorm         │
│  AveragePooling(2×2)            │
└─────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  Conv Block 3                   │
│  Conv2D(128) → BatchNorm        │
│  AveragePooling(2×2)            │
└─────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  Conv Block 4                   │
│  Conv2D(256) → BatchNorm        │
└─────────────────────────────────┘
       │
       ▼
GlobalAveragePooling2D
       │
       ▼
Dense(100) → Dense(70) → Dense(30) → Dense(15)
[Each block: BatchNorm + Dropout(0.3) + L2 Reg]
       │
       ▼
Dense(10) + Softmax
```

---

## ✨ Techniques Used

### 🔧 Regularization
| Technique | Purpose |
|-----------|---------|
| **L2 Regularization** (λ=0.0001) | Penalizes large weights |
| **Dropout (0.3)** | Randomly drops neurons during training |
| **Batch Normalization** | Stabilizes and speeds up training |

### 📈 Training Strategy
| Callback | Configuration |
|----------|--------------|
| **EarlyStopping** | Monitors `val_loss`, patience=10 |
| **ReduceLROnPlateau** | Halves LR on plateau, patience=3 |
| **Adam Optimizer** | Learning rate = 0.0005 |

### 🔄 Data Augmentation
```python
ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
```

---

## 📊 Results

<div align="center">

| Metric | Score |
|--------|-------|
| 🟢 **Training Accuracy** | ~89% |
| 🔵 **Testing Accuracy** | ~83% |

</div>

### Evaluation Plots Generated
- ✅ Accuracy vs Epochs (Train / Validation)
- ✅ Loss vs Epochs (Train / Validation)
- ✅ Confusion Matrix (10×10 heatmap)
- ✅ ROC Curves for all 10 classes (One-vs-Rest)

---

## 📁 Project Structure

```
Image-Classification-CNN/
│
├── 📓 notebooks/
│   └── Image_Classification_using_CNN.ipynb
│
├── 🖼️ assets/               # Saved plots / figures
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/GaneshGonge/Image-Classification-CNN.git
cd Image-Classification-CNN

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the notebook
jupyter notebook notebooks/Image_Classification_using_CNN.ipynb
```

> 💡 CIFAR-10 dataset downloads automatically — no manual setup needed!

---

## 🔮 Future Improvements

- [ ] Transfer Learning with **ResNet50**
- [ ] Transfer Learning with **MobileNet**
- [ ] Transfer Learning with **EfficientNet**
- [ ] **Real-time Image Prediction** via webcam or upload interface
- [ ] Deploy model as a **Flask / Streamlit web app**

---

## 🛠️ Tech Stack

| Tool | Usage |
|------|-------|
| Python | Core language |
| TensorFlow / Keras | Model building & training |
| NumPy | Array operations |
| Matplotlib & Seaborn | Visualization |
| Scikit-learn | Evaluation metrics |

---

## 👨‍💻 Author

<div align="center">

**Ganesh Gonge**
BSc IT Student | Machine Learning Enthusiast

[![GitHub](https://img.shields.io/badge/GitHub-GaneshGonge-181717?style=flat-square&logo=github)](https://github.com/GaneshGonge)

*If this project helped you, please consider giving it a ⭐ star — it means a lot!*

</div>
