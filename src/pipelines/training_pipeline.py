from src.entity.config_entity import DataIngestionConfig
from src.features_components.data_ingestion import DataIngestion
from src.logger import logger


def main():
    try:
        logger.info("Pipeline Execution Started")

        # Step 1: Data Ingestion
        config = DataIngestionConfig()
        data_ingestion = DataIngestion(config)
        artifact = data_ingestion.initiate_data_ingestion()

        logger.info("Pipeline Execution Completed Successfully")
        print("✅ Data Ingestion Completed")
        print("Raw Data:", artifact.raw_file_path)
        print("Train Data:", artifact.train_file_path)
        print("Test Data:", artifact.test_file_path)

    except Exception as e:
        logger.error(f"Pipeline Failed: {e}")
        raise e


def main():
    print("Training pipeline started...")

if __name__ == "__main__":
    main()


    