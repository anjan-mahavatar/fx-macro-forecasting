import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def train_price_predictor(X_train, y_train, X_test, y_test):
    print("Initializing Ensemble Method: LightGBM Regressor Stage...")
    
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbose': -1
    }
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[valid_data],
        num_boost_round=100
    )
    
    preds = model.predict(X_test)
    
    # CALCULATE CLEAN RMSE: Compute traditional MSE, then use numpy for the square root
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    
    print(f"LightGBM Training Complete. Test Set RMSE: {rmse:.4f}")
    return model, preds