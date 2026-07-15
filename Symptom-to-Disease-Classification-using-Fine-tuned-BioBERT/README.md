# 🧬 Symptom-to-Disease Classification using Fine-tuned BioBERT
![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Accuracy](https://img.shields.io/badge/Accuracy-97.50%25-green)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b)

## 📌 Project Overview
This project is an AI-powered Symptom-to-Disease Classification System developed using Natural Language Processing and Transformer-based Deep Learning. The system analyzes patient-written symptom descriptions (free text) and classifies them into 24 disease categories, ranging from skin conditions to infectious and chronic diseases.

The model is built by fine-tuning **BioBERT**, a domain-specific Transformer pretrained on biomedical literature, allowing it to better understand medical language compared to a general-purpose BERT model. Hyperparameter tuning (learning rate, weight decay, warmup steps) was applied to reduce overfitting and improve generalization on unseen symptom text.

An interactive Streamlit web application allows users to type in their symptoms and instantly receive a predicted disease category along with a confidence score.

---
## 🚀 Features
- Free-Text Symptom Classification (natural language input)
- Detection across 24 Disease Categories, including:
  - Psoriasis, Acne, Fungal Infection, Impetigo
  - Dengue, Malaria, Typhoid, Chicken pox
  - Bronchial Asthma, Pneumonia, Common Cold
  - Diabetes, Hypertension, Migraine, and more
- Fine-tuned BioBERT (Domain-Specific Transformer)
- Hyperparameter-Tuned Training (Learning Rate, Weight Decay, Warmup)
- Overfitting Diagnosis & Correction
- Confidence Score Prediction
- Interactive Streamlit Web Application
- Real-Time Symptom Analysis
---
## 🛠️ Technologies Used
| Technology | Usage |
|---|---|
| Python | Core Programming Language |
| PyTorch | Model Training Backend |
| Hugging Face Transformers | BioBERT Model & Tokenizer |
| BioBERT | Domain-Specific Transformer Backbone |
| Scikit-learn | Label Encoding & Evaluation Metrics |
| Pandas / NumPy | Data Handling |
| Matplotlib / Seaborn | Confusion Matrix Visualization |
| Streamlit | Web Application |
---
## 📊 Model Performance
| Metric | Score |
|---|---|
| Test Accuracy | **97.50%** |
| Train Accuracy | 100% |
| Weighted F1-Score | **0.97** |
| Macro F1-Score | 0.98 |
| Perfect-Score Classes | 19 / 24 |
---
## 📊 Workflow
1. Load & Clean Symptom-Disease Dataset
2. Encode Disease Labels (Text → Numbers)
3. Tokenize Symptom Text using BioBERT Tokenizer
4. Fine-tune BioBERT for 24-Class Classification
5. Diagnose Overfitting (Train vs Test Accuracy Gap)
6. Hyperparameter Tuning (Learning Rate, Weight Decay, Warmup)
7. Evaluate using Confusion Matrix & Classification Report
8. Deploy via Streamlit Web App
---
## 🚀 How to Run
```bash
# Clone repo
git clone https://github.com/ggonge16-star/symptom-to-disease-bert
# Install dependencies
pip install -r requirements.txt
# Run app
streamlit run app.py
```
---
## 📁 Dataset
- **Source:** Kaggle — Symptom2Disease Dataset
- **Classes:** 24 Diseases (Psoriasis, Dengue, Diabetes, Migraine, etc.)
- **Total Samples:** 1,200 symptom descriptions (50 per class, balanced)
- **Split:** 80% Train, 20% Test
---
## 🎯 Objective
The primary objective of this project is to demonstrate how Transformer-based NLP models can understand unstructured, natural-language symptom descriptions and map them to probable disease categories. The system aims to provide a fast, first-pass, explainable classification that could support (not replace) clinical triage and decision-making.
---
## 📷 Output
- Predicted Disease Category
- Confidence Percentage
- Classification Report (Precision, Recall, F1-Score)
- Confusion Matrix Visualization
---
## ⚠️ Limitations
- Trained on a relatively small dataset (1,200 samples); real-world deployment would require a larger, more linguistically diverse dataset.
- This is a research/portfolio project demonstrating NLP and Transformer fine-tuning skills — **it is not a validated medical diagnostic tool** and should not be used for actual clinical decision-making.
---
## 👨‍💻 Author
**Ganesh Gonge**
BSc IT | Machine Learning & Deep Learning Enthusiast
Passionate about AI, NLP, Computer Vision, and Real-World Deep Learning Applications ❤️🧬🚀
- 📧 ggonge16@gmail.com
- 🔗 [LinkedIn](https://www.linkedin.com/in/ganesh-gonge-369a24277/)
- 💻 [GitHub](https://github.com/ggonge16-star)
