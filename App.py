import streamlit as st
import pandas as pd
import joblib

# Loading model and scaler
model = joblib.load(r"C:\Users\WELCOME\PycharmProjects\FlipkartProject\Model\best_model.pkl")
scaler = joblib.load(r"C:\Users\WELCOME\PycharmProjects\FlipkartProject\Model\scaler.pkl")
feature_columns = joblib.load(r"C:\Users\WELCOME\PycharmProjects\FlipkartProject\Model\feature_columns.pkl")

# USER INPUT
st.set_page_config(page_title="Logistic Regression Prediction", layout="centered")
st.title("🔮 Washing Machine Prediction")

Capacity_kg = st.number_input("Capacity (kg)", 1.0, 20.0, step=0.5)
RPM = st.number_input("RPM", 500, 2000, step=50)
Price = st.number_input("Price", 5000, 100000, step=500)
Discount = st.number_input("Discount (%)", 0, 90, step=1)
Rating_Score = st.number_input("Rating Score", 0.0, 5.0, step=0.1)
Total_Ratings = st.number_input("Total Ratings", 0, 100000, step=10)
Total_Reviews = st.number_input("Total Reviews", 0, 100000, step=10)
load_type = st.selectbox("Load Type", ["Front Load", "Top Load"])
machine_type = st.selectbox("Machine Type", ["Semi Automatic", "Fully Automatic"])

# Encoding
load_encoded = 0 if load_type == "Front Load" else 1
machine_encoded = 0 if machine_type == "Semi Automatic" else 1

# PREDICTION
if st.button("Predict Cluster"):
    input_df = pd.DataFrame([{
        "Capacity_kg": Capacity_kg,
        "RPM": RPM,
        "Price": Price,
        "Discount": Discount,
        "Rating_Score": Rating_Score,
        "Total_Ratings": Total_Ratings,
        "Total_Reviews": Total_Reviews,
        "Load_Type": load_encoded,
        "Machine_Type": machine_encoded
    }])


    input_df = input_df[feature_columns]

    # Scaling and predicting
    input_scaled = scaler.transform(input_df)
    cluster = model.predict(input_scaled)[0]

    # Mapping cluster number to label
    cluster_map = {0: "Economy 💰", 1: "Standard ⭐", 2: "Premium 🏆"}
    st.success(f"✅ Predicted Cluster: **{cluster_map[cluster]}**")
    st.balloons()
