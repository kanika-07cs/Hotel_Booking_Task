import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
import joblib
scaler = pkl.load(open("scaler.pkl", "rb"))
pt = pkl.load(open("power_transformer.pkl", "rb"))
label_encoders = pkl.load(open("label_encoders.pkl", "rb"))  
feature_order = pkl.load(open("data_columns.pkl", "rb"))

rf_model = pkl.load(open("model_rf.pkl", "rb"))

xgb_model = pkl.load(open("model_xgb.pkl", "rb"))

num_cols = ['average price', 'total_guest', 'total_nights']
le_cols = ['booking_window_category'] 
basic_cols = ['total_guest','car parking space','repeated','P-C',
              'average price','special requests','total_nights','booking_window_category']
ohe_cols = [col for col in feature_order if col not in basic_cols]


st.title("Hotel Booking Status Prediction")
st.header("Enter Booking Details")

input_dict = {}
for col in basic_cols:
    if col in num_cols + ['special requests']:
        input_dict[col] = st.number_input(f"{col}", value=0.0)
    elif col in le_cols:
        options = list(label_encoders[col].classes_)
        selected = st.selectbox(f"{col}", options)
        input_dict[col] = selected
    else:  
        input_dict[col] = st.selectbox(f"{col}", [0,1])
cat_groups = {}
for col in ohe_cols:
    cat_name = col.split('_')[0]
    cat_value = '_'.join(col.split('_')[1:])
    if cat_name not in cat_groups:
        cat_groups[cat_name] = []
    cat_groups[cat_name].append(cat_value)

selected_cats = {}
for cat_name, options in cat_groups.items():
    selected = st.selectbox(f"{cat_name}", options)
    selected_cats[cat_name] = selected

for col in ohe_cols:
    cat_name = col.split('_')[0]
    cat_value = '_'.join(col.split('_')[1:])
    input_dict[col] = 1 if selected_cats.get(cat_name) == cat_value else 0

input_df = pd.DataFrame([input_dict])
for col in le_cols:
    le = label_encoders[col]
    input_df[col] = le.transform(input_df[[col]])
input_df[num_cols] = pt.transform(input_df[num_cols])
for col in feature_order:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[feature_order]
input_scaled = scaler.transform(input_df)

model_choice = st.selectbox("Choose Model", ["Random Forest", "XGBoost"])
model = rf_model if model_choice=="Random Forest" else xgb_model

threshold = 0.4  
if st.button("Predict"):
    pred_prob = model.predict_proba(input_scaled)[0][1]
    pred = 1 if pred_prob >= threshold else 0
    result = "Confirmed" if pred==1 else "Cancelled"
    st.subheader(f"Prediction: {result}")
    st.write(f"Probability of confirmation: {pred_prob:.2f}")
