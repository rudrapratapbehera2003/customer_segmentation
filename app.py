import streamlit as st
import pandas as pd
import numpy as np
import pickle

@st.cache_resource
def load_assets():
    with open('robust_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('pca_model.pkl', 'rb') as f:
        pca = pickle.load(f)
    with open('dbscan_predictor.pkl', 'rb') as f:
        model = pickle.load(f)
    mapping = pd.read_csv('cluster_mapping.csv')
    return scaler, pca, model, mapping

scaler, pca, model, mapping = load_assets()

st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("Customer Persona Predictor")

col1, col2 = st.columns(2)

with col1:
    education = st.selectbox("Education Level", ["Graduation", "PhD", "Master", "2n Cycle", "Basic"])
    marital = st.selectbox("Marital Status", ["Married", "Together", "Single", "Divorced", "Widow", "Alone", "Absurd", "YOLO"])
    income = st.number_input("Annual Income", min_value=0, value=20000)
    kidhome = st.number_input("Kids at Home", 0, 5, 0)
    teenhome = st.number_input("Teens at Home", 0, 5, 0)
    response = st.radio("Response", ["Yes", "No"],horizontal=True)
    complain = st.radio("Complain", ["Yes", "No"],horizontal=True)

with col2:
    recency = st.number_input("Recency (Days)", 0, 100, 30)
    mnt_wines = st.number_input("Wine Spend", 0, 2000, 100)
    mnt_fruits = st.number_input("Fruits Spend", 0, 500, 20)
    mnt_meat = st.number_input("Meat Spend", 0, 2000, 100)
    mnt_fish = st.number_input("Fish Spend", 0, 500, 20)
    mnt_sweet = st.number_input("Sweets Spend", 0, 500, 20)
    mnt_gold = st.number_input("Gold Spend", 0, 500, 20)

st.sidebar.header("Activity Data")
num_deals = st.sidebar.slider("Deals Purchases", 0, 20, 1)
num_web = st.sidebar.slider("Web Purchases", 0, 20, 2)
num_catalog = st.sidebar.slider("Catalog Purchases", 0, 20, 1)
num_store = st.sidebar.slider("Store Purchases", 0, 20, 3)
num_visits = st.sidebar.slider("Web Visits/Month", 0, 20, 5)

if st.button("Predict Customer Segment"):
    
    mar_divorced = 1 if marital == "Divorced" else 0
    mar_married = 1 if marital == "Married" else 0
    mar_single = 1 if marital in ["Single", "Alone", "Absurd", "YOLO"] else 0
    mar_together = 1 if marital == "Together" else 0
    mar_widow = 1 if marital == "Widow" else 0

    edu_val = 3 if education == "2n Cycle" else 0
    edu_val = 1 if education == "Basic" else 0
    edu_val = 2 if education == "Graduation" else 0
    edu_val = 3 if education == "Master" else 0
    edu_val = 4 if education == "PhD" else 0

    complain_val = 1 if complain == "Yes" else 0
    response_val = 1 if response == "Yes" else 0

    features = [
        edu_val,income, kidhome, teenhome, recency, mnt_wines, mnt_fruits, 
        mnt_meat, mnt_fish, mnt_sweet, mnt_gold, num_deals, num_web, 
        num_catalog, num_store, num_visits, 0, 0, 0, 0, 0, complain_val, response_val, 
        mar_divorced, mar_married, mar_single, mar_together, mar_widow
    ]
    
    input_df = pd.DataFrame([features], columns=scaler.feature_names_in_)
    
    scaled_data = scaler.transform(input_df)
    pca_data = pca.transform(scaled_data)
    cluster_id = model.predict(pca_data)[0]
    
    res = mapping[mapping['Cluster'] == cluster_id].iloc[0]
    
    st.divider()
    st.header(f"Result: {res['Persona']}")
    
    c1, c2 = st.columns(2)
    c1.metric("Top Category", res['Top Category'])
    c2.metric("Weakness", res['Key Weakness'])
    
    st.subheader("Marketing Strategy")
    st.info(res['Business Strategy'])