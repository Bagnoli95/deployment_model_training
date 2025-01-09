import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder

from utils.logger import log_line

import warnings
warnings.filterwarnings('ignore')

def process_raw_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
   """
   Procesa los datos:
   - Eliminar todas las filas donde existan valores nulos
   - Parsear la columna 'DT_COSTUMER' a tipo fecha

   Args:
      data (pd.DataFrame): DataFrame con los datos a procesar.

   Returns:
      pd.DataFrame: DataFrame con los datos procesados
   """
   # Eliminar todas las filas donde existan valores nulos
   df = df.dropna()
   log_line(f"✅ Total de filas luego de eliminar los valores nulos: {df.shape[0]}.")
   
   # Parsear la columna 'DT_COSTUMER' a tipo fecha
   df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y')
   log_line(f"✅ Tipos de Datos de las columna Dt_Customer despues de parsear: {df.dtypes[7]}.")
   
   return df


def process_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
   """
   Procesa los datos:
   Los datos están bastante limpios ahora y aprovechamos para incluir nuevas características.

   Args:
      data (pd.DataFrame): DataFrame con los datos a procesar.

   1. LabelEncoding para las características categoricas.
   2. StandarScaler para las demás características
   3. Crearemos un dataset para la reducción de dimensionalidad

   Returns:
      pd.DataFrame: DataFrame con los datos procesados
   """
   # Crearemos una lista de las variables categoricas
   categorical = df.select_dtypes(include=['object'])
   categorical_list = list(categorical.columns)

   log_line(f"✅ Variables Categoricas:\n{categorical_list}.")
   
   
# LabelEncoding para las variables categoricas
   label_encoder = LabelEncoder()
   for column in categorical_list:
      df[column] = label_encoder.fit_transform(df[column])

   log_line(f"✅ Todas las columnas son numericas ahora.")
   
   return df