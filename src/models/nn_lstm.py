import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

def train_volatility_nn(X_train, y_train, X_test, y_test):
    print("Initializing AI Stage: Neural Network Volatility Regressor...")
    
    nn_model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=200,
        random_state=42
    )
    
    nn_model.fit(X_train, y_train)
    preds = nn_model.predict(X_test)
    
    # CALCULATE CLEAN RMSE: Avoid squared parameter deprecation crashes
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    
    print(f"Neural Network Training Complete. Volatility RMSE: {rmse:.4f}")
    return nn_model, preds