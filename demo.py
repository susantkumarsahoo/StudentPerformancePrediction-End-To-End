from src.pipelines.training_pipeline import TrainingPipeline

dataset_path = r"C:\Users\LENOVO\MachineLearningProhects\StudentPerformancePrediction-End-To-End\data\raw\student.csv"
pipeline = TrainingPipeline(dataset_path=dataset_path)
pipeline.run_pipeline()