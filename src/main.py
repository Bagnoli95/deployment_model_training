import sys
import os

# Agregar la raíz del proyecto al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding='utf-8')

from utils.logger import configurar_logger, log_error, log_line, log_section

from src.data.data_loader import load_data
from src.data.data_processor import process_raw_data, process_data
from src.data.data_engineering import engineering_data
from data.data_outliers_handler import outlier_handler
from data.data_clustering import data_clustering
from src.data.data_splitter import split_data

from model.definition import create_model
from src.model.trainer import train_model
from src.model.evaluator import evaluate_model
from src.model.saver import save_model

def main():
    
    try:
    
        # Inicializar el logger
        configurar_logger()
        log_section("🚀 INICIO DE LA EJECUCIÓN DEL SCRIPT PRINCIPAL 🚀")
        
        # Cargar los datos
        log_section("CARGA DE DATOS")
        raw_data = load_data(file_path = "data/raw/marketing_campaign.csv")
        log_line(f"✅ Datos cargados exitosamente: {len(raw_data)} registros encontrados.")
        
        # Procesar los datos raw
        log_section("PROCESAMIENTO DE DATOS EN BRUTO")
        processed_raw_data = process_raw_data(df=raw_data)
        log_line("✅ Procesamiento de datos en bruto completado.")
        
        # Ingenieria de caracteristicas
        log_section("INGENIERÍA DE CARACTERÍSTICAS")
        eng_data = engineering_data(df=processed_raw_data)
        log_line("✅ Ingeniería de características completada.")
        
        # Manejo de Outliers
        log_section("MANEJO DE OUTLIERS")
        clear_data = outlier_handler(eng_data)
        log_line("✅ Manejo de outliers completado.")
        
        # Procesar los datos
        log_section("PROCESAMIENTO DE DATOS POST OUTLIERS")
        processed_data = process_data(clear_data)
        log_line("✅ Procesamiento de datos completado.")
        
        # Clusterizacion de datos
        log_section("CLUSTERIZACIÓN DE DATOS")
        cluster_data = data_clustering(df=processed_data)
        log_line("✅ Clusterización completada.")
        
        # Dividir los datos en conjuntos de entrenamiento y prueba
        log_section("DIVISIÓN DE DATOS")
        X_train, X_test, y_train, y_test = split_data(cluster_data)
        log_line("✅ Datos divididos exitosamente.")
        
        # Crear el modelo
        log_section("CREACIÓN DEL MODELO")
        model, X_train_scaled, X_test_scaled = create_model(X_train, X_test)
        log_line("✅ Modelo creado exitosamente.")
        
        # Entrenar el modelo
        log_section("ENTRENAMIENTO DEL MODELO")
        model = train_model(model, X_train_scaled, X_test_scaled, y_train, y_test)
        log_line("✅ Entrenamiento del modelo completado.")

        # Evaluar el modelo
        log_section("EVALUACIÓN DEL MODELO")
        loss, accuracy = evaluate_model(model, X_test_scaled, y_test)
        log_line(f"✅ Evaluación completada. Pérdida: {loss:.4f}, Precisión: {accuracy:.4f}")

        # Guardar el modelo
        log_section("GUARDADO DEL MODELO")
        full_path = save_model(model, model_path="models/trained_model")
        log_line(f"✅ Modelo guardado exitosamente en '{full_path}'.")
    
    except Exception as e:
        log_error("❌ Ocurrió un error durante la ejecución:", e)
        
    log_section("🎉 EJECUCIÓN DEL SCRIPT PRINCIPAL FINALIZADA 🎉")
    
if __name__ == "__main__":
    main()