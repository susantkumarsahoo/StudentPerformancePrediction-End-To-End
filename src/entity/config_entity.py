
from dataclasses import dataclass
from src.constants import constants

@dataclass
class DataIngestionConfig:
    raw_data_file: str = constants.RAW_DATA_FILE
    train_file: str = constants.TRAIN_FILE
    test_file: str = constants.TEST_FILE
    test_size: float = constants.TEST_SIZE
    random_state: int = constants.RANDOM_STATE
