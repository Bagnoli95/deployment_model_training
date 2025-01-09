import pandas as pd
import numpy as np

from utils.logger import log_line

def outlier_handler(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Procesa los datos:
    - Eliminar todas las filas donde existan valores nulos
    - Parsear la columna 'DT_COSTUMER' a tipo fecha

    Args:
        data (pd.DataFrame): DataFrame con los datos a procesar.

    Returns:
        pd.DataFrame: DataFrame con los datos procesados
    """
    #Eliminar outliers que no cumplan con los criterios de edad e ingresos 
    df = df[(df["Edad"]<90)]
    df = df[(df["Income"]<600000)]
    log_line(f"✅ El número total de rows del dataset luego de eliminar las columnas: {len(df)}.")
    
    return df