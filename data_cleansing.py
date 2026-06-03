import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_clean_data(filepath):
    # 1. Import
    df = pd.read_csv(filepath, parse_dates=['Date']).set_index('Date')
    
    # 2. Data Cleansing: Forward fill missing values to prevent data leakage
    df = df.fillna(method='ffill').dropna()
    
    # 3. Feature scaling (Crucial for Neural Networks)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[['Open', 'High', 'Low', 'Volume']])
    df_scaled = pd.DataFrame(scaled_features, columns=['Open', 'High', 'Low', 'Volume'], index=df.index)
    
    return df_scaled, scaler
