import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="centered"
)

st.title("🏦 Customer Churn Prediction App")

# =========================
# Load Model + Preprocessor
# =========================

@st.cache_resource
def load_assets():

    from tensorflow.keras.models import load_model

    model = load_model(r"E:\ANN\ann_model.keras")

    with open(r"E:\ANN\preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)

    return model, preprocessor

try:
    with st.spinner("Loading Model... ⏳"):
        model, preprocessor = load_assets()

    st.success("✅ Model Loaded Successfully")

except Exception as e:
    st.error(f"Error Loading Files : {e}")

# =========================
# User Inputs
# =========================

st.header("Enter Customer Details")

credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=650
)

geography = st.selectbox(
    "Geography",
    ["France", "Germany", "Spain"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

tenure = st.number_input(
    "Tenure",
    min_value=0,
    max_value=10,
    value=5
)

balance = st.number_input(
    "Balance",
    min_value=0.0,
    max_value=300000.0,
    value=50000.0
)

num_products = st.number_input(
    "Number Of Products",
    min_value=1,
    max_value=4,
    value=1
)

has_card = st.selectbox(
    "Has Credit Card",
    [0, 1]
)

is_active = st.selectbox(
    "Is Active Member",
    [0, 1]
)

salary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    max_value=300000.0,
    value=100000.0
)

# =========================
# Prediction
# =========================

if st.button("Predict"):

    try:

        input_df = pd.DataFrame({

            'CreditScore': [credit_score],
            'Geography': [geography],
            'Gender': [gender],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_products],
            'HasCrCard': [has_card],
            'IsActiveMember': [is_active],
            'EstimatedSalary': [salary]

        })

        # Preprocess Input
        transformed_data = preprocessor.transform(input_df)

        # Prediction
        prediction = model.predict(transformed_data)

        prediction_value = prediction[0][0]

        st.subheader("Prediction Result")

        if prediction_value > 0.5:
            st.error("⚠️ Customer Will Leave The Bank")
        else:
            st.success("✅ Customer Will Stay")

        st.write(f"Prediction Score : {prediction_value:.4f}")

    except Exception as e:
        st.error(f"Prediction Error : {e}")