import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def train_lightgbm(X_train, y_train, X_test, y_test):
    # LightGBM is highly optimized for tabular data and handles large datasets efficiently
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    params = {'objective': 'regression', 'metric': 'rmse', 'learning_rate': 0.05}
    
    model = lgb.train(params, train_data, valid_sets=[valid_data], num_boost_round=100)
    
    preds = model.predict(X_test)
    print("LightGBM RMSE:", mean_squared_error(y_test, preds, squared=False))
    return model
