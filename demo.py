# from src.pipelines.training_pipeline import TrainingPipeline

#if __name__ == "__main__":
    #dataset_path = "C:\\Users\\TPWODL\\New folder_Content\\StudentPerformancePrediction-End-To-End\\data\\raw\\student.csv"   # put your dataset path here
    #pipeline = TrainingPipeline(dataset_path=dataset_path)
    ##pipeline.run_pipeline()


# main.py
'''
from src.logging.logger import get_logger
from src.exceptions.exception import CustomException
import sys

logger = get_logger(__name__)

def divide(a, b):
    try:
        return a / b
    except Exception as e:
        raise CustomException(str(e), sys)
    
# Logging should be outside the function
logger.info("Starting ML pipeline...")

try:
    result = divide(10, 0)
    logger.info(f"Result: {result}")
except CustomException as ce:
    logger.error(ce)
'''
from src.pipelines.training_pipeline import TrainingPipeline

dataset_path = "C:\\Users\\TPWODL\\New folder_Content\\StudentPerformancePrediction-End-To-End\\data\\raw\\student.csv"   # put your dataset path here
pipeline = TrainingPipeline(dataset_path=dataset_path)
pipeline.run_pipeline()