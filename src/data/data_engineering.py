import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

def engineering_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Agregar la ingeniería de características al conjunto de datos:

    Returns:
        pd.DataFrame: DataFrame con los datos procesados
    """
    #Age of customer today 
    df["Edad"] = 2014-df["Year_Birth"]

    #Total spendings on various items
    df["Gastos"] = df["MntWines"]+ df["MntFruits"]+ df["MntMeatProducts"]+ df["MntFishProducts"]+ df["MntSweetProducts"]+ df["MntGoldProds"]

    #Deriving living situation by marital status"Alone"
    df["En_Convivenvia"]=df["Marital_Status"].replace({"Married":"Partner", "Together":"Partner", "Absurd":"Alone", "Widow":"Alone", "YOLO":"Alone", "Divorced":"Alone", "Single":"Alone",})

    #Feature indicating total children living in the household
    df["Hijos"]=df["Kidhome"]+df["Teenhome"]

    #Feature for total members in the householde
    df["Tamanho_familiar"] = df["En_Convivenvia"].replace({"Alone": 1, "Partner":2})+ df["Hijos"]

    #Feature pertaining parenthood
    df["Es_Padre"] = np.where(df.Hijos> 0, 1, 0)

    #Segmenting education levels in three groups
    df["Education"]=df["Education"].replace({"Basic":"Undergraduate","2n Cycle":"Undergraduate", "Graduation":"Graduate", "Master":"Postgraduate", "PhD":"Postgraduate"})

    #For clarity
    df=df.rename(columns={"MntWines": "Wines","MntFruits":"Fruits","MntMeatProducts":"Meat","MntFishProducts":"Fish","MntSweetProducts":"Sweets","MntGoldProds":"Gold"})

    #Dropping some of the redundant features
    to_drop = ["Marital_Status", "Dt_Customer", "Z_CostContact", "Z_Revenue", "Year_Birth", "ID"]
    df = df.drop(to_drop, axis=1)

    return df