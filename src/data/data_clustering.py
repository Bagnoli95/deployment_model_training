import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler

def data_clustering(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Procesa los datos:
    - Eliminar todas las filas donde existan valores nulos
    - Parsear la columna 'DT_COSTUMER' a tipo fecha

    Args:
        data (pd.DataFrame): DataFrame con los datos a procesar.

    Returns:
        pd.DataFrame: DataFrame con los datos procesados
        """
    # Creamos una copia del DataSet
    df_copy = df.copy()

    # Creamos un subgrupo de las columnas que no queremos estandarizar
    no_escalar = ['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1','AcceptedCmp2', 'Complain', 'Response']
    df_copy = df_copy.drop(no_escalar, axis=1)

    #Escalamos
    scaler = StandardScaler()
    scaler.fit(df_copy)
    df_scalado = pd.DataFrame(scaler.transform(df_copy), columns=df_copy.columns)

    print('DataFrame Escalado.')
    
    #Inicialiar el PCA para disminuir la dimensionalidad
    pca = PCA(n_components=3)
    pca.fit(df_scalado)
    PCA_ds = pd.DataFrame(pca.transform(df_scalado), columns=(["col1","col2", "col3"]))
    
    # Iniciar el modelo
    AC = AgglomerativeClustering(n_clusters=4)
    # fit model and predict clusters
    yhat_AC = AC.fit_predict(PCA_ds)
    PCA_ds["Clusters"] = yhat_AC
    #Adding the Clusters feature to the orignal dataframe.
    df["Clusters"]= yhat_AC
    
    # Crear una nueva columna "target" basada en un umbral de gastos
    threshold = 500
    df['target'] = (df['Gastos'] > threshold).astype(int)
    
    
    return df