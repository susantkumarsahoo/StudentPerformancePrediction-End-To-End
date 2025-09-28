from src.pipelines.training_pipeline import TrainingPipeline
from src.pipelines.deployment_pipeline import DeploymentPipeline

# Correct dataset path
dataset_path = r"C:\Users\TPWODL\New folder_Content\StudentPerformancePrediction-End-To-End\data\raw\student.csv"

# Initialize and run the training pipeline
deployment_pipeline = DeploymentPipeline(dataset_path=dataset_path)
deployment_pipeline.run_deployment_pipeline()




