# ✈️ Customer Booking Prediction – British Airways

![Python](https://img.shields.io/badge/Python-3.10-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-brightgreen)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.x-orange)
![F1 Score](https://img.shields.io/badge/F1--Score-0.43-yellow)
![Recall](https://img.shields.io/badge/Recall-60%25-red)
![Forage](https://img.shields.io/badge/Forage-British%20Airways-blue)

## 📌 Project Overview

This project is a Machine Learning based Customer Booking Prediction System developed as part of the **British Airways Data Science Job Simulation** on Forage. The system analyzes customer flight booking data and predicts whether a customer will **complete a booking or not**.

The model is built using **XGBoost** with a fully custom **Scikit-learn preprocessing pipeline**, handling severe class imbalance (85:15) through `scale_pos_weight` rather than synthetic oversampling. To improve business usability, **feature importance analysis** is used to identify which factors most influence a customer's decision to book, and the findings are summarized in an executive PowerPoint slide.

---

## 🚀 Features

- Customer Booking Completion Prediction (Binary Classification)
- End-to-End Preprocessing Pipeline:
  - Skewness Correction (Log Transformation)
  - Outlier Capping (IQR Method)
  - Custom Frequency Encoder for high-cardinality columns
- Feature Engineering (5 new engineered features)
- Class Imbalance Handling (SMOTE vs `scale_pos_weight` comparison)
- Multi-Model Benchmarking (KNN, Random Forest, Decision Tree, Gradient Boosting, XGBoost)
- Hyperparameter Tuning with RandomizedSearchCV
- Overfitting Diagnosis (Train vs Test comparison)
- Threshold Tuning
- Feature Importance Visualization
- Executive PowerPoint Summary Reporting

---

## 🛠️ Technologies Used

| Technology | Usage |
|---|---|
| Python | Core Programming Language |
| Pandas / NumPy | Data Handling & Numerical Computing |
| Scikit-learn | Preprocessing Pipeline, Modeling, Evaluation |
| XGBoost | Final Predictive Model |
| imbalanced-learn (SMOTE) | Class Imbalance Handling (comparison) |
| Matplotlib | Feature Importance Visualization |
| Jupyter Notebook / Google Colab | Development Environment |
| PowerPoint | Business Reporting |

---

## 📊 Model Performance

| Metric | Train | Test |
|---|---|---|
| F1-Score | 0.50 | **0.43** |
| Precision | 0.39 | 0.34 |
| Recall | 0.70 | **0.60** |
| Accuracy | – | **0.76** |

**Top Predictors (Feature Importance):** Booking Origin, Sales Channel (Mobile), Wants Extra Baggage, Route, Flight Duration.

### 🆚 Comparison vs Forage's Official Reference Solution

| Metric | Official Reference | This Project |
|---|---|---|
| Precision | 0.70 | 0.34 |
| **Recall** | **0.003** | **0.60** |
| Class Imbalance Handled? | ❌ No | ✅ Yes (`scale_pos_weight`) |

The official reference model correctly identified only ~3 out of 1,000 actual bookers (essentially defaulting to "always predict no booking"). By explicitly addressing the 85:15 class imbalance, this project's model catches **60% of real bookers** — making it significantly more useful for proactive customer targeting.

---

## 📊 Workflow

1. Load Customer Booking Dataset (50,000+ records)
2. Exploratory Data Analysis (Univariate, Bivariate, Correlation)
3. Data Cleaning — Skewness Correction & Outlier Capping
4. Feature Engineering (5 new features)
5. Preprocessing Pipeline (Scaling, Frequency Encoding, One-Hot Encoding)
6. Class Imbalance Handling (SMOTE → `scale_pos_weight`)
7. Multi-Model Benchmarking (5-Fold Stratified Cross-Validation)
8. Hyperparameter Tuning (RandomizedSearchCV)
9. Overfitting Check (Train vs Test) & Threshold Tuning
10. Feature Importance Visualization
11. Executive PowerPoint Summary & Model Export

---

## 🚀 How to Run

```bash
# Clone repo
git clone https://github.com/ggonge16-star/customer-booking-prediction

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook Customer_Booking_Prediction_British_Airways.ipynb
```

---

## 📁 Dataset

- **Source:** Provided by British Airways via Forage (Data Science Job Simulation, Task 2)
- **Target:** `booking_complete` (0 = No Booking, 1 = Booking Completed)
- **Total Records:** 50,000+ customer bookings
- **Class Distribution:** ~85% No Booking, ~15% Booking Completed
- **Split:** 80% Train, 20% Test (Stratified)

---

## 🎯 Objective

The primary objective of this project is to help British Airways proactively identify customers who are likely to complete a booking, rather than reacting only once a customer arrives at the airport. The system aims to provide accurate, imbalance-aware predictions and interpretable insights into what drives a customer's decision to book.

---

## 🧠 Challenges Faced & How I Solved Them

- **Severe class imbalance (85:15):** Default models barely predicted the minority class (F1 as low as 0.17–0.22). Solved by testing SMOTE first, then switching to XGBoost's `scale_pos_weight`, which performed better.
- **High-cardinality categorical columns** (`route`: 799 values, `booking_origin`: 104 values): One-Hot Encoding would have created hundreds of sparse columns. Solved by building a custom Frequency Encoder transformer compatible with Scikit-learn pipelines.
- **ColumnTransformer silently dropping engineered columns:** Learned that any column not explicitly listed in the transformer gets dropped by default — fixed by ensuring every new feature was added to the correct column list.
- **Overfitting after initial tuning:** An unconstrained search picked `max_depth=8`, causing a noticeable train-test gap. Re-tuned with tighter regularization, reducing the gap from 0.136 to 0.070.
- **Diminishing returns in tuning:** Recognized when further hyperparameter search gave marginal gains (<0.01 F1), indicating a ceiling in the dataset's predictive signal rather than continuing to over-optimize.

---

## 📷 Output

- Booking Completion Prediction (0/1)
- Precision, Recall, F1-Score & Classification Report
- Feature Importance Bar Chart (Top 15 Predictors)
- Confusion Matrix
- Executive PowerPoint Summary Slide

---

## 📁 Repository Contents

- `Customer_Booking_Prediction_British_Airways.ipynb` — Full notebook (EDA → Cleaning → Feature Engineering → Modeling → Evaluation)
- `customer_booking.csv` — Raw dataset
- `Cleaned_data_Customer_Booking_British_Airways.csv` — Cleaned dataset
- `Customer_Booking_Prediction_Summary.pptx` — Executive summary slide
- `XGBoost_Customer_Booking_Model.pkl` — Final saved model pipeline

---

## 🎓 Certification

This project was completed as part of the **British Airways Data Science Job Simulation** on [Forage](https://www.theforage.com).

---

## 👨‍💻 Author

**Ganesh Gonge**
BSc IT | Machine Learning & Deep Learning Enthusiast
Passionate about AI, Data Science, and Real-World Machine Learning Applications ❤️📊🚀

- 📧 ggonge16@gmail.com
- 🔗 [LinkedIn](https://www.linkedin.com/in/ganesh-gonge-369a24277/)
- 💻 [GitHub](https://github.com/ggonge16-star)
