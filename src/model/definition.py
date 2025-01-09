import pandas as pd
from sklearn.discriminant_analysis import StandardScaler

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

import warnings
warnings.filterwarnings('ignore')

def create_model(X_train:pd.DataFrame, X_test:pd.DataFrame):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Definir el modelo MLP
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),  # Capa oculta 1
        Dense(32, activation='relu'),                                         # Capa oculta 2
        Dense(1, activation='sigmoid')                                       # Capa de salida
    ])
        
    # Compilar el modelo
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
    
    return model, X_train_scaled, X_test_scaled