# Symptom-to-Disease Classification using Fine-tuned BioBERT

A Natural Language Processing project that classifies diseases from patient-written symptom descriptions using a fine-tuned BioBERT (domain-specific Transformer) model.

## Overview

This project takes free-text symptom descriptions (e.g., *"I have been experiencing joint pain in my fingers and a silver-like dusting on my skin"*) and predicts the most likely disease out of 24 categories. Unlike traditional keyword-matching systems, this model uses a Transformer-based architecture to understand the semantic meaning and context of symptom descriptions.

## Problem Statement

Manually mapping unstructured symptom text to a disease category is time-consuming and inconsistent. This project automates that process using a fine-tuned biomedical language model, providing a fast, explainable, first-pass classification that could support (not replace) clinical decision-making.

## Dataset

- **Source:** [Symptom2Disease](https://www.kaggle.com/datasets/niyarrbarman/symptom2disease) (Kaggle)
- **Size:** 1,200 samples across 24 disease categories (50 samples per class, balanced)
- **Columns:** `text` (symptom description), `label` (disease name)

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Model | BioBERT (`dmis-lab/biobert-base-cased-v1.1`) |
| Framework | Hugging Face Transformers, PyTorch |
| Data Handling | Pandas, NumPy, Scikit-learn |
| Evaluation | Confusion Matrix, Classification Report (Precision/Recall/F1) |
| Visualization | Matplotlib, Seaborn |
| Deployment | Streamlit |

## Approach

1. **Data Preprocessing** — Cleaned the dataset (null/duplicate checks) and encoded disease labels using `LabelEncoder`.
2. **Tokenization** — Used the BioBERT tokenizer with `max_length=128`, padding, and truncation to convert symptom text into model-ready tensors.
3. **Model Fine-tuning** — Fine-tuned `BertForSequenceClassification` (BioBERT weights) on the training split using Hugging Face's `Trainer` API.
4. **Hyperparameter Tuning** — Tuned learning rate, weight decay, and warmup steps to reduce overfitting and improve generalization.
5. **Evaluation** — Assessed performance using accuracy, weighted F1-score, per-class classification report, and confusion matrix analysis.
6. **Deployment** — Built a Streamlit interface for real-time symptom-to-disease prediction.

## Results

| Metric | Score |
|---|---|
| Test Accuracy | **97.50%** |
| Weighted F1-score | **0.97** |
| Macro F1-score | 0.98 |

- 19 out of 24 disease categories achieved a perfect (1.00) F1-score.
- Hyperparameter tuning improved test accuracy from 92.92% to 97.50% (~5% gain) and reduced the train-test accuracy gap from ~5.7% to ~2.5%.
- Confusion matrix analysis revealed clinically coherent misclassifications (e.g., Dengue vs. Chicken pox, Peptic Ulcer Disease vs. GERD), where real-world symptom overlap explains the model's errors.

## Project Structure

```
├── Symptom2Disease.csv          # Dataset
├── symptom_disease_bert.ipynb   # Main notebook (preprocessing, training, evaluation)
├── biobert_symptom_model/       # Saved fine-tuned model + tokenizer + label encoder
├── app.py                       # Streamlit web application
└── README.md
```

## How to Run

```bash
# Clone the repository
git clone <your-repo-link>
cd symptom-to-disease-bert

# Install dependencies
pip install transformers torch scikit-learn pandas numpy matplotlib seaborn streamlit

# Run the Streamlit app
streamlit run app.py
```

## Limitations

- Trained on a relatively small dataset (1,200 samples); real-world deployment would require a larger and more linguistically diverse dataset for robust generalization.
- This is a research/portfolio project intended to demonstrate NLP and Transformer fine-tuning skills — **it is not a validated medical diagnostic tool** and should not be used for actual clinical decision-making.

## Future Improvements

- Expand the dataset with more diverse, real-world phrased symptom descriptions.
- Add attention-weight visualization for explainable predictions (highlighting which words drove the prediction).
- Experiment with other biomedical Transformers (e.g., ClinicalBERT, PubMedBERT) for comparison.

## Author

**Ganesh Gonge**
📧 ggonge16@gmail.com | [LinkedIn](#) | [GitHub](#)
