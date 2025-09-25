import os
import sys
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataPreprocessingArtifact
from src.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataPreprocessingConfig
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_preprocessing import DataPreprocessing


class TrainingPipeline:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    def run_pipeline(self):
        # Step 1: Data Ingestion
        ingestion_config = DataIngestionConfig(dataset_path=self.dataset_path)
        ingestion = DataIngestion(ingestion_config)
        ingestion_artifact = ingestion.initiate_data_ingestion()

        # Step 2: Data Validation
        validation_config = DataValidationConfig()
        validation = DataValidation(
            data_ingestion_artifact=ingestion_artifact,
            data_validation_config=validation_config
        )
        validation_artifact = validation.initiate_data_validation()

        # Step 3: Data Preprocessing
        preprocessing_config = DataPreprocessingConfig()
        preprocessing = DataPreprocessing(
            data_ingestion_artifact=ingestion_artifact,
            data_validation_artifact=validation_artifact,
            data_preprocessing_config=preprocessing_config
        )
        preprocessing_artifact = preprocessing.initiate_data_preprocessing()

