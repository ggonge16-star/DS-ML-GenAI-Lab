# 🤖 AI Job Risk Prediction using Machine Learning

## 📌 Overview

This project predicts the risk of jobs being impacted by Artificial Intelligence (AI).
It classifies jobs into three categories:

* 🟢 Low Risk
* 🟡 Medium Risk
* 🔴 High Risk

---

## 🧠 Problem Statement

With the rapid growth of AI, many jobs are at risk of automation.
The goal of this project is to build a machine learning model that predicts AI risk levels based on job-related features.

---

## 📊 Dataset

The dataset contains both categorical and numerical features such as:

* Job Title
* Primary Skill
* Salary
* Salary Bucket

Target Variable:

* `ai_risk_category`

---

## 🧹 Data Preprocessing

* Removed irrelevant and redundant features
* Handled categorical data using **One-Hot Encoding**
* Performed feature selection using:

  * Chi-Square Test
  * Cramér’s V
  * Correlation Analysis

---

## ⚙️ Models Used

* Logistic Regression
* Support Vector Machine (SVM)
* Random Forest
* Gradient Boosting

---

## 🏆 Final Model

**GradientBoostingClassifier** was selected as the best model because:

* Highest F1-score (~0.84)
* Balanced performance across all classes
* No overfitting (Train ≈ Test)

---

## 🔧 Hyperparameter Tuning

Used **RandomizedSearchCV** to optimize:

* n_estimators
* learning_rate
* max_depth
* subsample

---

## 📈 Evaluation Metrics

* F1 Score
* Accuracy
* Confusion Matrix
* ROC-AUC

---

## ✅ Results

* F1 Score: ~0.84
* Model shows strong generalization
* Accurate prediction of high-risk jobs

---

## ⚖️ Overfitting Check

Train and test scores are very close, indicating a balanced and reliable model.

---

## 🛠️ Tech Stack

* Python
* Scikit-learn
* XGBoost
* Pandas, NumPy
* Matplotlib, Seaborn

---

## 🚀 Future Improvements

* Feature Engineering
* Ensemble Learning
* Model Deployment

---

## 💡 Conclusion

This project demonstrates a complete machine learning workflow, from data preprocessing to model tuning and evaluation, to predict AI job risk effectively.

---
