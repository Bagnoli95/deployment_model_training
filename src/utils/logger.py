import logging

cantCaracter = 80

def configurar_logger():
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,  # Nivel de logs
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.StreamHandler(),  # Mostrar logs en consola
            logging.FileHandler("execution.log", mode="w", encoding="utf-8")  # Guardar logs en un archivo
        ]
    )
    
def log_section(title):
    """Imprime una sección destacada en el log."""
    separator = "=" * cantCaracter
    logging.info(separator)
    logging.info(f"{title.center(cantCaracter)}")
    # logging.info(separator + "\n")
    
def log_line(line):
    """Imprime una sección normal en el log."""
    logging.info(line + "\n")
    
def log_error(title, error):
    """Imprime una sección con error en el log."""
    separator = "=" * cantCaracter
    logging.error(separator)
    logging.error(f"{title.center(cantCaracter)}")
    # logging.error(separator + "\n")
    logging.error (error)



