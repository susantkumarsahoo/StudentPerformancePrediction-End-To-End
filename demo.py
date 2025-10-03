import os
import sys
from src.pipelines.deployment_pipeline import DeploymentPipeline
from src.pipelines.prediction_pipeline import StudentDataInput
from src.logging.logger import get_logger
from src.exceptions.exception import CustomException
from src.constants.constants import*

logger = get_logger(__name__)


def main():
    try:
        logger.info("=" * 80)
        logger.info("🚀 STARTING STUDENT PERFORMANCE PREDICTION SYSTEM")
        logger.info("=" * 80)
        
        # Step 1: Find dataset path
        logger.info("🔍 Locating dataset...")
        
        path1 = DATA_PATH_01
        path2 = DATA_PATH_02
        
        if os.path.exists(path1):
            dataset_path = path1
            logger.info(f"✅ Dataset found at: {path1}")
        elif os.path.exists(path2):
            dataset_path = path2
            logger.info(f"✅ Dataset found at: {path2}")
        else:
            raise FileNotFoundError("❌ student.csv not found in either path!")

        # Step 2: Run Deployment Pipeline (Training)
        logger.info("\n" + "=" * 80)
        logger.info("📊 PHASE 1: MODEL TRAINING AND DEPLOYMENT")
        logger.info("=" * 80)
        
        deployment_pipeline = DeploymentPipeline(dataset_path=dataset_path)
        deployment_pipeline.run_deployment_pipeline()
        
        logger.info("✅ Model trained and deployed successfully!")

        # Step 3: Test Prediction with Sample Data
        logger.info("\n" + "=" * 80)
        logger.info("🎯 PHASE 2: TESTING PREDICTION")
        logger.info("=" * 80)
        
        # Create sample student data for testing
        sample_student = StudentDataInput(
            gender="female",
            race_ethnicity="group B",
            parental_level_of_education="bachelor's degree",
            lunch="standard",
            test_preparation_course="none",
            reading_score=72,
            writing_score=74
        )
        
        logger.info(f"📝 Sample Student Data:")
        logger.info(f"   Gender: {sample_student.gender}")
        logger.info(f"   Race/Ethnicity: {sample_student.race_ethnicity}")
        logger.info(f"   Parental Education: {sample_student.parental_level_of_education}")
        logger.info(f"   Lunch Type: {sample_student.lunch}")
        logger.info(f"   Test Prep Course: {sample_student.test_preparation_course}")
        logger.info(f"   Reading Score: {sample_student.reading_score}")
        logger.info(f"   Writing Score: {sample_student.writing_score}")
        
        # Make prediction
        predicted_score = deployment_pipeline.make_prediction(sample_student)
        
        logger.info("\n" + "=" * 80)
        logger.info(f"🎯 PREDICTED MATH SCORE: {predicted_score:.2f}")
        logger.info("=" * 80)
        
        print("\n" + "=" * 80)
        print(f"✅ SUCCESS! Predicted Math Score: {predicted_score:.2f}")
        print("=" * 80)
        
        logger.info("\n🎉 System executed successfully!")
        
    except Exception as e:
        logger.error("❌ CRITICAL ERROR IN MAIN EXECUTION", exc_info=True)
        print(f"\n❌ Error: {str(e)}")
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()

















