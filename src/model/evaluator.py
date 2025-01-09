from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score,roc_curve
import pandas as pd
from tensorflow.keras import Sequential

from sklearn.metrics import classification_report

import warnings

from utils.logger import log_line
warnings.filterwarnings('ignore')

def evaluate_model(model: Sequential,
                X_test_scaled:pd.DataFrame,
                y_test:pd.Series):
    
    """
    Función para evaluar el modelo
    
    Args:
        model (xgboost.sklearn.XGBClassifier): Modelo a evaluar
        test_data (pd.DataFrame): Datos de prueba
        y_test (pd.Series): Etiquetas de prueba

    Returns:
        tuple[float, float, float, float, float]: Accuracy, Precision, Recall, F1, AUC
    """
    # Evaluación del modelo en el conjunto de prueba
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)

    # Predicciones y matriz de confusión
    y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
    
    # Informe de clasificación
    log_line(f"✅ Informe de Clasificación:")
    print(classification_report(y_test, y_pred, target_names=["No Compra", "Compra"]))
    
    return loss, accuracy
