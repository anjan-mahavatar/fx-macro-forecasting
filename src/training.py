import sys
import os
# Ensure python recognizes the root project directory paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_import import import_external_datasets
from src.data_cleansing import clean_and_prepare_features
from src.models.ml_lightgbm import train_price_predictor
from src.models.nn_lstm import train_volatility_nn

# 1. Run Ingestion Pipeline
raw_data = import_external_datasets(start_date="2015-01-01", end_date="2025-01-01")

# 2. Run Cleansing and Feature Extraction
cleaned_df, feature_cols = clean_and_prepare_features(raw_data)

# 3. Create Time-Series Aware Chronological Split (80% Train / 20% Test)
X = cleaned_df[feature_cols + ['dayofweek', 'month']]
y_price = cleaned_df['Target_Price']
y_vol = cleaned_df['Target_Volatility']

split_idx = int(len(cleaned_df) * 0.8)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train_p, y_test_p = y_price.iloc[:split_idx], y_price.iloc[split_idx:]
y_train_v, y_test_v = y_vol.iloc[:split_idx], y_vol.iloc[split_idx:]

# 4. Train Model 1 (Ensemble Model for Pricing)
price_model, price_preds = train_price_predictor(X_train, y_train_p, X_test, y_test_p)

# 5. Train Model 2 (Neural Network for Volatility)
vol_model, vol_preds = train_volatility_nn(X_train, y_train_v, X_test, y_test_v)

print("\n🎉 SUCCESS: All Model Training Pipeline Stages Completed!")