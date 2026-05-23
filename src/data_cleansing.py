import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def clean_and_prepare_features(df):
    print("Executing structural data cleansing pipeline...")
    
    # Forward-fill weekends/holidays to prevent data leakage, then backfill remaining gaps
    df = df.ffill().bfill()
    
    # Feature Engineering
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['FX_returns'] = df['FX_Close'].pct_change().fillna(0)
    df['Oil_Momentum'] = df['Oil_Close'].pct_change(30).fillna(0)
    
    # Establish Dual Targets
    # Target 1: Next-day absolute closing price
    df['Target_Price'] = df['FX_Close'].shift(-1)
    # Target 2: Options-ready Realized Volatility (30-day rolling standard deviation)
    df['Target_Volatility'] = df['FX_returns'].rolling(window=30).std().shift(-30)
    
    # Drop rows broken by shifting forward/backward
    df = df.dropna()
    
    # Scale numeric features to prevent gradient issues in Neural Networks
    scaler = StandardScaler()
    feature_cols = ['FX_Close', 'Oil_Close', 'Gold_Close', 'DGS2', 'VIXCLS', 'FX_returns', 'Oil_Momentum']
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    return df, feature_cols