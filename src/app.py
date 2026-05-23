import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Align local paths so the app can talk to your existing modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_import import import_external_datasets
from src.data_cleansing import clean_and_prepare_features
from src.models.ml_lightgbm import train_price_predictor
from src.models.nn_lstm import train_volatility_nn

# Set up page configurations
st.set_page_config(page_title="FX Macro Forecasting Lab", layout="wide")

st.title("📊 FX Price & Volatility Forecasting Endpoint")
st.write("Level 7 Program Portfolio: Multi-Modal Machine Learning Sandbox")

# -------------------------------------------------------------------------
# CACHED DATA & TRAINING WORKFLOW (Runs silently on launch)
# -------------------------------------------------------------------------
@st.cache_resource
def initialize_and_train_pipeline():
    """Ingests data and trains models once, caching them to keep the UI snappy."""
    # Pull a targeted timeframe for fast web rendering
    raw_data = import_external_datasets(start_date="2020-01-01", end_date="2025-01-01")
    cleaned_df, feature_cols = clean_and_prepare_features(raw_data)
    
    all_features = feature_cols + ['dayofweek', 'month']
    X = cleaned_df[all_features]
    y_p = cleaned_df['Target_Price']
    y_v = cleaned_df['Target_Volatility']
    
    split_idx = int(len(cleaned_df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    
    # Train our core engines
    price_model, _ = train_price_predictor(X_train, y_p.iloc[:split_idx], X_test, y_p.iloc[split_idx:])
    vol_model, _ = train_volatility_nn(X_train, y_v.iloc[:split_idx], X_test, y_v.iloc[split_idx:])
    
    return price_model, vol_model, all_features, X_test, y_p.iloc[split_idx:], y_v.iloc[split_idx:]

with st.spinner("Initializing live API datasets and training background architectures..."):
    price_model, vol_model, feature_names, X_test, y_test_p, y_test_v = initialize_and_train_pipeline()

# -------------------------------------------------------------------------
# SIDEBAR CONTROL INTERFACE
# -------------------------------------------------------------------------
st.sidebar.header("🕹️ Live Macro Scenario Simulator")
st.sidebar.write("Modify real-time baseline values to simulate market shifts.")

# Create input fields based on our engineered features
fx_close = st.sidebar.slider("Current FX Rate (EUR/USD Standardized)", -3.0, 3.0, 0.0)
oil_close = st.sidebar.slider("Brent Crude Oil Shock Index", -3.0, 3.0, 0.5)
gold_close = st.sidebar.slider("Gold Spot Safe-Haven Scale", -3.0, 3.0, -0.2)
dgs2_rate = st.sidebar.slider("2-Year Treasury Yield Shift", -3.0, 3.0, 0.1)
vix_index = st.sidebar.slider("CBOE VIX Volatility Panic Gauge", -3.0, 3.0, 1.2)
fx_returns = st.sidebar.number_input("Prior Asset Return Matrix", value=0.0)
oil_mom = st.sidebar.number_input("30-Day Crude Momentum Vector", value=0.02)
day_of_week = st.sidebar.selectbox("Day of Week Execution", [0, 1, 2, 3, 4], format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri"][x])
month_of_year = st.sidebar.slider("Execution Month Spectrum", 1, 12, 5)

# Gather user inputs into a structured vector DataFrame matching our models
user_input_vector = pd.DataFrame([[
    fx_close, oil_close, gold_close, dgs2_rate, vix_index, 
    fx_returns, oil_mom, day_of_week, month_of_year
]], columns=feature_names)

# -------------------------------------------------------------------------
# INTERACTIVE INFERENCE & PLOTTING DASHBOARD
# -------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔮 Model Inference Engine")
    if st.button("Trigger Dual-Model Live Inference"):
        # Run live cross-model inference
        predicted_price = price_model.predict(user_input_vector)[0]
        predicted_vol = vol_model.predict(user_input_vector)[0]
        
        # Display professional result cards
        st.metric(label="Model 1: Predicted Next-Day Pricing Index", value=f"{predicted_price:.4f}")
        st.metric(label="Model 2: Predicted Options Volatility Index", value=f"{predicted_vol:.4f}")
        
        st.success("Inference metrics cleanly generated via endpoints.")

with col2:
    st.subheader("📈 Time-Series Performance Graph")
    # Generate an evaluation chart showing how our Ensemble model tracks the test data
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(y_test_p.index[-40:], y_test_p.values[-40:], label="Actual Historic Track", color="black", linewidth=1.5)
    
    # Simulate a baseline forward line using test inferences
    simulated_preds = price_model.predict(X_test.iloc[-40:])
    ax.plot(y_test_p.index[-40:], simulated_preds, label="LightGBM Ensemble Inference", color="blue", linestyle="--", linewidth=1.5)
    
    ax.set_title("Historical Out-of-Sample Performance Testing")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)