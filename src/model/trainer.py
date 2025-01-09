import xgboost as xgb
import pandas as pd

import tensorflow as tf
from tensorflow.keras import Sequential

def train_model(model: Sequential,
                X_train_scaled:pd.Series,
                X_test_scaled:pd.Series,
                y_train:pd.Series,
                y_test:pd.Series):
    """
    Función para entrenar el modelo
    """
    
    model.fit(X_train_scaled, y_train, 
                    validation_data=(X_test_scaled, y_test),
                    epochs=50, batch_size=32, verbose=0)
    
    return model