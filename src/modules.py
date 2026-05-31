# modules.py
import numpy as np
import scipy.stats as si
import tensorflow as tf
from tensorflow.keras import Sequential, layers

def create_3d_sequences(data, steps):
    """
    Transforms flat 2D historical vectors into 3D sequential array matrices.
    Expected structural dimension: [Batch_Size, Time_Steps, Features]
    """
    X_arr, y_arr = [], []
    for i in range(len(data) - steps):
        X_arr.append(data[i:(i + steps), 0])
        y_arr.append(data[i + steps, 0])
    X_arr = np.array(X_arr)
    y_arr = np.array(y_arr)
    return np.reshape(X_arr, (X_arr.shape[0], X_arr.shape[1], 1)), y_arr

def build_network(architecture, lookback_window):
    """
    Compiles Keras neural network configurations dynamically matching user preferences.
    Uses explicit Input layers to avoid Keras 3 dimension locking errors.
    """
    model = Sequential()
    model.add(tf.keras.Input(shape=(lookback_window, 1)))
    
    if architecture == "Simple RNN":
        model.add(layers.SimpleRNN(32, activation='tanh'))
    elif architecture == "LSTM":
        model.add(layers.LSTM(50, activation='tanh'))
    elif architecture == "CNN-LSTM Network":
        model.add(layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.LSTM(32, activation='tanh'))
    elif architecture == "Graph Interaction Proxy (GCN-V)":
        model.add(layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='same'))
        model.add(layers.Flatten())
        model.add(layers.Dense(32, activation='relu'))
        
    # Standard output layer for prediction regression tasks
    model.add(layers.Dense(1)) 
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def calculate_fx_option_price(S, K, T, r_d, r_f, sigma, option="Call Option"):
    """
    Garman-Kohlhagen analytical pricing model for foreign exchange options.
    Handles dual interest rate boundaries (domestic and foreign).
    """
    d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option == "Call Option":
        price = (S * np.exp(-r_f * T) * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r_d * T) * si.norm.cdf(d2, 0.0, 1.0))
    else:
        price = (K * np.exp(-r_d * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * np.exp(-r_f * T) * si.norm.cdf(-d1, 0.0, 1.0))
    return price