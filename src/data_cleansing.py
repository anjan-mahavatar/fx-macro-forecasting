import pandas as pd
from sklearn.preprocessing import StandardScaler

def clean_and_prepare_features(df):
    df = df.ffill().bfill()
    
    # Feature Engineering with Lags
    df['FX_return_lag_1'] = df['FX_Close'].pct_change().shift(1)
    df['VIX_lag_1'] = df['VIXCLS'].shift(1)
    df['Oil_Momentum_lag_5'] = df['Oil_Close'].pct_change(5).shift(1)
    
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    
    df['Target_Price'] = df['FX_Close'].shift(-1)
    df['Target_Volatility'] = df['FX_Close'].pct_change().rolling(30).std().shift(-30)
    
    df = df.dropna()
    
    feature_cols = ['FX_Close', 'Oil_Close', 'Gold_Close', 'DGS2', 'VIXCLS', 
                    'FX_return_lag_1', 'VIX_lag_1', 'Oil_Momentum_lag_5', 'dayofweek', 'month']
    
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    return df, feature_cols