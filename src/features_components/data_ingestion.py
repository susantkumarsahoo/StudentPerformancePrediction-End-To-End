import os
import pandas as pd
from sklearn.model_selection import train_test_split

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.logger import logger


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        logger.info("Starting Data Ingestion Process")

        os.makedirs(os.path.dirname(self.config.raw_data_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.config.train_file), exist_ok=True)

        # Dummy dataset for demo (replace with actual CSV later)
        df = pd.DataFrame({
            "feature1": range(1, 101),
            "feature2": [x * 2 for x in range(1, 101)],
            "target": [x % 2 for x in range(1, 101)]
        })

        df.to_csv(self.config.raw_data_file, index=False)
        logger.info(f"Raw data saved at {self.config.raw_data_file}")

        # Train-Test Split
        train_set, test_set = train_test_split(
            df,
            test_size=self.config.test_size,
            random_state=self.config.random_state
        )

        train_set.to_csv(self.config.train_file, index=False)
        test_set.to_csv(self.config.test_file, index=False)

        logger.info("Data split completed")
        logger.info(f"Train data saved at {self.config.train_file}")
        logger.info(f"Test data saved at {self.config.test_file}")

        return DataIngestionArtifact(
            train_file_path=self.config.train_file,
            test_file_path=self.config.test_file,
            raw_file_path=self.config.raw_data_file
        )
