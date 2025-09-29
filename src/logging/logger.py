import logging
import os
from datetime import datetime
from src.constants.constants import*
from src.entity.model_entity import LoggingConfig
from src.entity.artifact_entity import LoggingArtifact


# Create logs directory if not exists
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Dynamic log file name with timestamp
LOG_FILE = os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log")

# Configure logging
logging.basicConfig(
    filename=LOG_FILE,
    format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO
)

# Get logger instance
def get_logger(module_name: str) -> logging.Logger:
    """
    Returns a logger instance with the given module name.
    """
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)

    # Stream Handler (prints logs to console as well)
    if not logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s - %(name)s - %(message)s")
        )
        logger.addHandler(stream_handler)

    return logger















