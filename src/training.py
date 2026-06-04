import sys
import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
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
# Ensure this matches your renamed/updated module
from src.models.nn_lstm import train_volatility_nn 

def evaluate_models_tscv(X, y, n_splits=5):
    """
    Evaluates baseline models using MLflow-tracked Walk-Forward Validation.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    models = {
        'Ridge': Ridge(alpha=1.0),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5),
        'LightGBM': LGBMRegressor(n_estimators=100, learning_rate=0.05, num_leaves=31)
    }

    results = []

    for name, model in models.items():
        fold_metrics = []
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            with mlflow.start_run(run_name=f"Baseline_{name}_Fold_{fold}", nested=True):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                
                rmse = np.sqrt(mean_squared_error(y_test, preds))
                mlflow.log_metric("rmse", rmse)
                fold_metrics.append(rmse)
        
        results.append({'Model': name, 'Mean RMSE': np.mean(fold_metrics)})
    
    return pd.DataFrame(results).sort_values(by='Mean RMSE')

if __name__ == "__main__":
    mlflow.set_experiment("FX_Forecasting_L7_Project")
    print("🚀 Starting Integrated Pipeline with MLflow...")

    # 1 & 2. Ingestion & Cleansing
    raw_data = import_external_datasets()
    cleaned_df, feature_cols = clean_and_prepare_features(raw_data)

    X = cleaned_df[feature_cols]
    y_price = cleaned_df['Target_Price']
    y_vol = cleaned_df['Target_Volatility']

    split_idx = int(len(cleaned_df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train_p, y_test_p = y_price.iloc[:split_idx], y_price.iloc[split_idx:]
    y_train_v, y_test_v = y_vol.iloc[:split_idx], y_vol.iloc[split_idx:]

    # 4. Production Price Predictor
    with mlflow.start_run(run_name="Production_Price_Predictor"):
        price_model, price_preds = train_price_predictor(X_train, y_train_p, X_test, y_test_p)
        mlflow.log_metric("test_rmse", np.sqrt(mean_squared_error(y_test_p, price_preds)))
        mlflow.sklearn.log_model(price_model, "price_model")
        print("✅ Price Predictor Logged to MLflow.")

    # 5. Production Volatility NN (True LSTM)
    with mlflow.start_run(run_name="Production_Volatility_LSTM"):
        vol_model, vol_preds = train_volatility_nn(X_train, y_train_v, X_test, y_test_v)
        mlflow.log_metric("test_rmse", np.sqrt(mean_squared_error(y_test_v, vol_preds)))
        mlflow.tensorflow.log_model(vol_model, "volatility_model")
        print("✅ Volatility LSTM Logged to MLflow.")

    # 6. Baseline Comparison (Walk-Forward)
    print("\n📊 Running Baseline Walk-Forward Validation...")
    comparison_df = evaluate_models_tscv(X, y_price, n_splits=5)
    print(comparison_df.to_markdown(index=False))

    print("\n🎉 SUCCESS: All Pipeline Stages Logged to MLflow!")