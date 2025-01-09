from sklearn.model_selection import train_test_split
import pandas as pd

def split_data(data:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Dividir los datos en conjuntos de entrenamiento y prueba

    Args:
        data (pd.DataFrame): Dataframe que contiene los datos a dividir
        target_column (str): Nombre de la columna objetivo
        test_size (float, optional): Proporción de los datos que quedan en el test. Defaults to 0.2.
        random_state (int, optional): Semilla para la aleatoriedad. Defaults to 42.

    Returns:
        Tupla: Una tupla que contiene los conjuntos de entrenamiento y prueba
        - X_train (pd.DataFrame): Conjunto de entrenamiento de las variables independientes
        - X_test (pd.DataFrame): Conjunto de prueba de las variables independientes
        - y_train (pd.Series): Conjunto de entrenamiento de la variable objetivo
        - y_test (pd.Series): Conjunto de prueba de la variable objetivo
    """
    
    X = data.drop(columns=['target', 'Gastos'])  # Eliminamos la columna target y Gastos del dataset
    y = data['target']
    
    # Dividir en conjunto de entrenamiento y prueba (80%-20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Confirmar las dimensiones de los conjuntos generados
    X_train.shape, X_test.shape, y_train.shape, y_test.shape
        
    return X_train, X_test, y_train, y_test