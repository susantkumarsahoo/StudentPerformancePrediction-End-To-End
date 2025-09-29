from src.pipelines.training_pipeline import TrainingPipeline
from src.pipelines.deployment_pipeline import DeploymentPipeline

# Correct dataset path
dataset_path = r"C:\Users\LENOVO\MachineLearningProhects\StudentPerformancePrediction-End-To-End\data\raw\student.csv"

# Initialize and run the training pipeline

'''
deployment_pipeline = DeploymentPipeline(dataset_path=dataset_path)
deployment_pipeline.run_deployment_pipeline()

'''

pipeline = TrainingPipeline(dataset_path=dataset_path)
pipe = pipeline.run_pipeline()





