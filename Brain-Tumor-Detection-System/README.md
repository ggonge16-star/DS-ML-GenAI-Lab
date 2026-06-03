# 🧠 Brain Tumor Detection & Visualization System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-95.19%25-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b)

## 📌 Project Overview
This project is an AI-powered Brain Tumor Detection and Visualization System developed using Deep Learning techniques. The system analyzes Brain MRI scans and automatically classifies them into four categories: Glioma, Meningioma, Pituitary Tumor, and No Tumor.

The model is built using MobileNetV2 and TensorFlow, achieving high classification accuracy while maintaining efficient performance. To improve transparency and explainability, Grad-CAM visualization is integrated to highlight the regions that influenced the model's prediction.

An interactive Streamlit web application allows users to upload MRI images, receive tumor predictions, view confidence scores, and visualize suspected tumor regions through heatmaps and contour localization.

---

## 🚀 Features
- Brain MRI Image Classification
- Detection of:
  - Glioma Tumor
  - Meningioma Tumor
  - Pituitary Tumor
  - No Tumor
- Deep Learning using MobileNetV2
- Grad-CAM Explainable AI Visualization
- Tumor Region Localization with OpenCV Contours
- Confidence Score Prediction
- Interactive Streamlit Web Application
- Real-Time MRI Analysis
- MRI Image Validation

---

## 🛠️ Technologies Used
| Technology | Usage |
|---|---|
| Python | Core Programming Language |
| TensorFlow / Keras | Model Building & Training |
| MobileNetV2 | Transfer Learning Backbone |
| OpenCV | Image Preprocessing & Contour Detection |
| NumPy | Numerical Computing |
| Streamlit | Web Application |
| Grad-CAM | Explainable AI Visualization |

---

## 📊 Model Performance
| Metric | Score |
|---|---|
| Test Accuracy | **95.19%** |
| Val Accuracy | **97%** |
| Precision (avg) | 0.95 |
| Recall (avg) | 0.95 |
| F1-Score (avg) | 0.95 |

---

## 📊 Workflow
1. Upload Brain MRI Scan
2. Image Preprocessing (OpenCV)
3. Deep Learning Model Prediction (MobileNetV2)
4. Tumor Classification
5. Confidence Score Calculation
6. Grad-CAM Heatmap Generation
7. Tumor Region Visualization (OpenCV Contours)
8. Final Diagnostic Output

---

## 🚀 How to Run
```bash
# Clone repo
git clone https://github.com/ggonge16-star/brain-tumor-detection

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

---

## 📁 Dataset
- **Source:** Kaggle — Brain Tumor MRI Dataset
- **Classes:** Glioma, Meningioma, No Tumor, Pituitary
- **Total Images:** 7000+ MRI scans
- **Split:** 80% Train, 20% Validation, Test set

---

## 🎯 Objective
The primary objective of this project is to assist in the early detection of brain tumors by leveraging Artificial Intelligence and Deep Learning techniques. The system aims to provide fast, accurate, and interpretable predictions from MRI scans while enhancing medical image analysis through visual explanations.

---

## 📷 Output
- Tumor Type Prediction
- Confidence Percentage
- Grad-CAM Heatmap
- Tumor Localization Contours
- MRI Visualization Dashboard

---

## 👨‍💻 Author
**Ganesh Gonge**
BSc IT | Machine Learning & Deep Learning Enthusiast
Passionate about AI, Medical Imaging, Computer Vision, and Real-World Deep Learning Applications ❤️🧠🚀

- 📧 ggonge16@gmail.com
- 🔗 [LinkedIn](https://www.linkedin.com/in/ganesh-gonge-369a24277/)
- 💻 [GitHub](https://github.com/ggonge16-star)
