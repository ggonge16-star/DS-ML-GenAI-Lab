# Customer Booking Prediction – British Airways ✈️

**Forage Virtual Internship — British Airways Data Science Job Simulation**
*Completed: July 2026*

A complete end-to-end machine learning project that predicts whether a customer will complete a flight booking, built as part of the British Airways Data Science virtual internship on Forage.

---

## 📌 Project Background

British Airways provided this task as part of their Data Science job simulation. The business problem: customers today have access to huge amounts of information and often research extensively before buying. This means airlines can no longer afford to be *reactive* (waiting until a customer reaches the airport) — they need to be **proactive**, identifying which customers are likely to complete a booking so they can be targeted early with the right offers.

**Goal:** Build a predictive model that classifies whether a customer will complete a booking (`booking_complete = 1`) or not (`booking_complete = 0`), and identify which factors influence that decision the most.

---

## 📊 Data Source

- Dataset: `customer_booking.csv`, provided directly by British Airways through the Forage platform (Data Science Job Simulation, Task 2).
- **50,000 rows, 14 columns** — includes passenger count, sales channel, trip type, purchase lead time, length of stay, flight timing/route/duration, add-on preferences (baggage, seat, meals), and the target column `booking_complete`.
- Target distribution was **imbalanced: ~85% did not complete a booking, ~15% did.**

---

## 🛠️ What I Did (Step by Step)

### 1. Exploratory Data Analysis (EDA)
- Checked shape, data types, duplicates, and missing values.
- Ran univariate analysis (histograms/boxplots) on numeric columns and countplots on categorical columns.
- Ran bivariate analysis of each feature against the target.
- Plotted a correlation heatmap to check for multicollinearity and relationships with the target.

### 2. Data Cleaning
- Found high positive skew in `num_passengers`, `purchase_lead`, and `length_of_stay` (skew > 1.5).
- Applied `log1p()` transformation to correct skew.
- Detected outliers using the IQR method; applied **capping (winsorization)** instead of deleting rows, to correct outliers without losing data.

### 3. Feature Engineering
Created 5 new features to give the model more signal to work with:
- `total_extras` — sum of baggage/seat/meal preferences (engagement proxy)
- `is_weekend` — whether the flight falls on Sat/Sun
- `flight_hour_category` — bucketed flight hour into Morning/Afternoon/Evening/Night
- `is_round_trip` — simplified trip type flag
- `is_last_minute` — flag for short purchase-lead bookings

### 4. Preprocessing Pipeline
Built a `ColumnTransformer`-based pipeline:
- **Numeric features** → `StandardScaler`
- **High-cardinality categoricals** (`route`, `booking_origin` — 799 and 104 unique values respectively) → a **custom Frequency Encoder** (built from scratch as a `BaseEstimator`/`TransformerMixin`), since One-Hot Encoding would have created hundreds of sparse columns and Label Encoding would have implied a false numeric order.
- **Low-cardinality categoricals** (`sales_channel`, `trip_type`, `flight_day`, `flight_hour_category`) → `OneHotEncoder`

### 5. Handling Class Imbalance
- First tried **SMOTE** oversampling inside the pipeline (via `imblearn.Pipeline`) — result was underwhelming.
- Switched to **XGBoost's built-in `scale_pos_weight` parameter**, which handles imbalance through loss-weighting instead of synthetic samples — this gave a significantly better result than SMOTE.

### 6. Model Comparison
Benchmarked 5 algorithms using **Stratified 5-Fold Cross-Validation** with `scoring='f1'` (F1 was chosen over accuracy because the target is imbalanced, and accuracy would be misleading — a model that always predicts "no booking" would already score ~85% accuracy while being useless):

| Model | F1 Score (CV) |
|---|---|
| KNN | 0.377 |
| Random Forest (tuned) | 0.361 |
| Decision Tree | 0.280 |
| Gradient Boosting (tuned) | 0.276 |
| **XGBoost (tuned + `scale_pos_weight`)** | **0.438** ✅ |

### 7. Hyperparameter Tuning
- Used `RandomizedSearchCV` (5-fold CV, `scoring='f1'`) across multiple rounds — first with SMOTE, then with `scale_pos_weight`, then with a regularization-focused search (lower `max_depth`, higher `min_child_weight`/`gamma`) to reduce overfitting.
- Final tuned parameters: `n_estimators=300, max_depth=6, learning_rate=0.03, min_child_weight=7, gamma=0.2, subsample=0.7, colsample_bytree=0.6, scale_pos_weight=4`.

### 8. Model Validation
- Compared train vs. test metrics to check for overfitting — regularization reduced the train–test F1 gap from **0.136 → 0.070**.
- Ran threshold tuning (testing cutoffs from 0.20–0.70) to confirm the default 0.5 threshold was already optimal, since `scale_pos_weight` had already balanced the model's output probabilities.

### 9. Feature Importance & Reporting
- Extracted `feature_importances_` from the final XGBoost model and visualized the top 15 predictors.
- Summarized the full pipeline, results, and business recommendation in a single executive PowerPoint slide (as required by the task brief), aimed at a non-technical manager audience.

---

## 📈 Final Results

| Metric | Train | Test |
|---|---|---|
| F1 Score | 0.50 | **0.43** |
| Precision | 0.39 | 0.34 |
| Recall | 0.70 | **0.60** |
| Accuracy | – | 0.76 |

**Top predictors of booking completion:** `booking_origin`, `sales_channel` (Mobile), `wants_extra_baggage`, `route`, `flight_duration`.

### 🆚 Comparison with Forage's Official Reference Solution
Forage provides an official "example answer" after submission. Comparing the two:

| | Official Reference Solution | This Project |
|---|---|---|
| Precision | 0.70 | 0.34 |
| **Recall** | **0.003** | **0.60** |
| Class imbalance handled? | No | Yes (`scale_pos_weight`) |

The reference solution's 0.3% recall means it correctly identified only ~3 out of 1,000 actual bookers — effectively defaulting to "always predict no booking." By explicitly handling the 85:15 class imbalance, this project's model catches **60% of real bookers**, making it far more useful for the stated business goal of proactively targeting likely customers.

---

## 🧠 Challenges Faced & How I Solved Them

- **Severe class imbalance (85/15 split):** Initial models (default RandomForest/XGBoost) barely predicted the minority class, giving F1 scores as low as 0.17–0.22. Solved by testing SMOTE first, then switching to `scale_pos_weight`, which worked better for tree-based boosting.
- **High-cardinality categorical columns** (`route`: 799 values, `booking_origin`: 104 values): One-Hot Encoding would have exploded dimensionality and caused sparse, hard-to-learn features. Solved by writing a custom Frequency Encoder transformer compatible with scikit-learn pipelines.
- **`ColumnTransformer` silently dropping engineered columns:** Learned that any column not explicitly listed in the transformer's column lists gets dropped by default — had to make sure every new engineered feature was added to the correct list (numeric/frequency/nominal).
- **Overfitting after initial tuning:** A shallower, unconstrained hyperparameter search picked `max_depth=8`, which showed a noticeable train–test gap. Re-tuned with tighter regularization (`max_depth=6`, higher `min_child_weight`/`gamma`) to close the gap while keeping test performance stable.
- **Diminishing returns:** After several tuning rounds, F1 improvements became marginal (~0.01 or less per round) — recognized this as a sign that the available features had a hard ceiling on predictive power, rather than continuing to over-optimize.

---

## 🧰 Tech Stack / Skills Used

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `XGBoost` · `imbalanced-learn (SMOTE)` · `Matplotlib` · `Jupyter Notebook` · `Google Colab` · `PowerPoint` (executive reporting)

**Concepts:** EDA, Skewness Correction, Outlier Treatment (IQR/Winsorization), Feature Engineering, Custom Scikit-learn Transformers, ColumnTransformer Pipelines, Class Imbalance Handling, Stratified K-Fold Cross-Validation, Hyperparameter Tuning (RandomizedSearchCV), Overfitting Diagnosis, Threshold Tuning, Feature Importance Analysis, Business Communication.

---

## 📂 Repository Contents

- `Customer_Booking_Prediction_British_Airways.ipynb` — full notebook (EDA → cleaning → feature engineering → modeling → evaluation)
- `customer_booking.csv` — raw dataset
- `Cleaned_data_Customer_Booking_British_Airways.csv` — cleaned dataset after preprocessing
- `Customer_Booking_Prediction_Summary.pptx` — executive summary slide
- `XGBoost_Customer_Booking_Model.pkl` — final saved model pipeline

---

## 🎓 Certification

This project was completed as part of the **British Airways Data Science Job Simulation** on [Forage](https://www.theforage.com).
