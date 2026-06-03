import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Ensure python recognizes the root project directory paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_import import import_external_datasets
from src.data_cleansing import clean_and_prepare_features
from src.models.ml_lightgbm import train_price_predictor
from src.models.nn_lstm import train_volatility_nn


def evaluate_models_tscv(X, y, n_splits=5):
    """
    Evaluates multiple regression models using Time Series Cross-Validation.
    Designed for continuous FX forecasting targets (Target_Price or Target_Volatility).
    """
    # Define the chronological cross-validation strategy
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Initialize models (using your existing LightGBM parameters as a baseline)
    models = {
        'Ridge Regression': Ridge(alpha=1.0),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, objective='reg:squarederror'),
        'LightGBM': LGBMRegressor(n_estimators=100, learning_rate=0.05, num_leaves=31, objective='regression')
    }

    results = []

    for name, model in models.items():
        fold_rmse = []
        fold_mae = []
        fold_dir_acc = []

        for train_index, test_index in tscv.split(X):
            # Ensure data remains in chronological order
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Train the model
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            # Calculate standard error metrics
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mae = mean_absolute_error(y_test, preds)

            # Calculate Directional Accuracy
            # Compares the sign of the predicted change vs actual change from the previous step
            actual_diff = np.diff(y_test)
            pred_diff = preds[1:] - y_test.iloc[:-1].values
            
            # Compute matching signs (Up vs Down)
            correct_direction = np.sum(np.sign(actual_diff) == np.sign(pred_diff))
            dir_acc = correct_direction / len(actual_diff) if len(actual_diff) > 0 else 0

            fold_rmse.append(rmse)
            fold_mae.append(mae)
            fold_dir_acc.append(dir_acc)

        # Aggregate results across all folds
        results.append({
            'Model': name,
            'Mean RMSE': np.mean(fold_rmse),
            'Mean MAE': np.mean(fold_mae),
            'Mean Directional Accuracy': np.mean(fold_dir_acc)
        })

    # Return a clean DataFrame, sorted by the primary performance metric
    return pd.DataFrame(results).sort_values(by='Mean RMSE')


if __name__ == "__main__":
    print("🚀 Starting Pipeline...")

    # 1. Run Ingestion Pipeline
    print("Importing datasets...")
    raw_data = import_external_datasets(start_date="2015-01-01", end_date="2025-01-01")

    # 2. Run Cleansing and Feature Extraction
    print("Cleansing and preparing features...")
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
    print("Training production price predictor...")
    price_model, price_preds = train_price_predictor(X_train, y_train_p, X_test, y_test_p)

    # 5. Train Model 2 (Neural Network for Volatility)
    print("Training production volatility neural network...")
    vol_model, vol_preds = train_volatility_nn(X_train, y_train_v, X_test, y_test_v)

    # 6. Advanced Time-Series Cross-Validation (Baseline Comparison)
    print("\n📊 Running Time-Series Cross-Validation for Baseline Comparison (Target: Price)...")
    price_comparison_df = evaluate_models_tscv(X, y_price, n_splits=5)
    
    print("\nModel Comparison Results:")
    print(price_comparison_df.to_markdown(index=False))

    print("\n🎉 SUCCESS: All Model Training Pipeline Stages Completed!")

```