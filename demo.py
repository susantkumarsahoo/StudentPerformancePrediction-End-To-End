from src.pipelines.training_pipeline import TrainingPipeline

# Correct dataset path
dataset_path = r"C:\Users\LENOVO\MachineLearningProhects\StudentPerformancePrediction-End-To-End\data\raw\student.csv"

# Initialize and run the training pipeline
pipeline = TrainingPipeline(dataset_path=dataset_path)
pipeline.run_pipeline()




