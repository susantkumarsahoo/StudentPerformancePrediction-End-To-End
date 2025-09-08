from src.entity.config_entity import DataIngestionConfig
from src.components.data_ingestion import DataIngestion

class TrainingPipeline:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    def run_pipeline(self):
        # Step 1: Data Ingestion
        ingestion_config = DataIngestionConfig(dataset_path=self.dataset_path)
        ingestion = DataIngestion(ingestion_config)
        ingestion_artifact = ingestion.initiate_data_ingestion()

        print("✅ Data Ingestion Completed")
        print(ingestion_artifact)

        # (Later you can add Data Validation, Transformation, Model Training)
