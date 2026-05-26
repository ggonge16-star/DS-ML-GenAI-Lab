<div align="center">

# 📉 Customer Churn Prediction Using ANN

A production-ready **Artificial Neural Network** built with TensorFlow/Keras to predict customer churn — helping businesses retain customers before they leave.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

**86%+ Validation Accuracy &nbsp;|&nbsp; Binary Classification &nbsp;|&nbsp; Class Weight Balancing**

</div>

---

## 📌 Overview

Customer churn is one of the most critical business problems — losing a customer costs **5× more** than retaining one. This project builds an end-to-end deep learning pipeline to predict whether a customer will churn, giving businesses the power to act before it's too late.

### What this project covers:
- Complete **data preprocessing pipeline** with encoding + scaling
- Custom **ANN architecture** with regularization techniques
- **Class weight balancing** to handle imbalanced churn data
- Full **evaluation suite** — accuracy, confusion matrix, classification report

---

## 📂 Dataset

**Source:** Customer Churn Dataset (Telecom)

| Feature | Description |
|---------|-------------|
| Gender | Customer gender |
| Senior Citizen | Whether customer is a senior citizen |
| Tenure | Number of months with the company |
| Monthly Charges | Monthly billing amount |
| Total Charges | Total amount billed |
| Contract Type | Month-to-month / One year / Two year |
| Payment Method | Electronic / Mailed check / Bank transfer |
| Internet Service | DSL / Fiber optic / No |
| **Churn** | **Target — Yes / No** |

---

## 🏗️ ML Workflow

```
Raw Data
    │
    ▼
┌─────────────────────────────────────┐
│  Data Preprocessing                 │
│  Missing values → OneHot Encoding   │
│  Standard Scaling → CT Pipeline     │
└─────────────────────────────────────┘
    │
    ▼
Train / Test Split
    │
    ▼
┌─────────────────────────────────────┐
│  ANN Architecture                   │
│  Dense → BatchNorm → Dropout        │
│  Dense → BatchNorm → Dropout        │
│  Output → Sigmoid                   │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Training                           │
│  Adamax Optimizer                   │
│  Binary Crossentropy Loss           │
│  EarlyStopping + Class Weights      │
└─────────────────────────────────────┘
    │
    ▼
Evaluation → Accuracy | Confusion Matrix | Classification Report
```

---

## ✨ Techniques Used

### 🔧 Preprocessing Pipeline
| Step | Method |
|------|--------|
| Categorical Encoding | OneHotEncoder via ColumnTransformer |
| Feature Scaling | StandardScaler |
| Pipeline | Scikit-learn Pipeline (no data leakage) |

### 🧠 ANN Architecture
| Layer | Detail |
|-------|--------|
| Input | All customer features |
| Hidden Layers | Dense + BatchNormalization + Dropout |
| Activation | ReLU (hidden), Sigmoid (output) |
| Output | 1 neuron → Churn probability |

### 🛡️ Regularization & Training
| Technique | Purpose |
|-----------|---------|
| Dropout | Prevents co-adaptation of neurons |
| Batch Normalization | Stabilizes and speeds up learning |
| EarlyStopping | Stops when val_loss plateaus |
| Class Weight Balancing | Handles imbalanced churn labels |
| Adamax Optimizer | Robust adaptive learning rate |

---

## 📊 Results

<div align="center">

| Metric | Score |
|--------|-------|
| 🟢 **Validation Accuracy** | **86%+** |
| 📋 Classification Report | Per-class Precision, Recall, F1 |
| 🔲 Confusion Matrix | True/False Positives & Negatives |

</div>

### Evaluation Outputs
- ✅ Training vs Validation Accuracy curve
- ✅ Training vs Validation Loss curve
- ✅ Confusion Matrix heatmap
- ✅ Full Classification Report

---

## 📁 Project Structure

```
Customer-Churn-Prediction-Using-ANN/
│
├── 📓 Ann_project.ipynb            # Main notebook — full pipeline
├── 📊 cleaned_Customer Churn.csv   # Preprocessed dataset
├── requirements.txt                # Python dependencies
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
git clone https://github.com/ggonge16-star/Customer-Churn-Prediction-Using-ANN.git
cd Customer-Churn-Prediction-Using-ANN

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the notebook
jupyter notebook Ann_project.ipynb
```

---

## 🔮 Future Improvements

- [ ] ROC-AUC Curve visualization
- [ ] Hyperparameter tuning with Keras Tuner
- [ ] Cross Validation (Stratified K-Fold)
- [ ] Model saving & loading (`model.save()`)
- [ ] Deploy as web app using **Flask / Streamlit**
- [ ] SHAP values for model explainability

---

## 🛠️ Tech Stack

| Tool | Usage |
|------|-------|
| Python | Core language |
| TensorFlow / Keras | ANN model building & training |
| Scikit-learn | Preprocessing pipeline & evaluation |
| Pandas & NumPy | Data manipulation |
| Matplotlib & Seaborn | Visualization |

---

## 👨‍💻 Author

<div align="center">

**Ganesh Babasaheb Gonge**
BSc IT Graduate · Data Science & ML Engineer

[![GitHub](https://img.shields.io/badge/GitHub-ggonge16--star-181717?style=flat-square&logo=github)](https://github.com/ggonge16-star)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Ganesh%20Gonge-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/ganesh-gonge)

*Found this useful? Drop a ⭐ — it keeps me motivated to build more!*

</div>
