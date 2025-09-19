from src.pipelines.training_pipeline import TrainingPipeline

dataset_path = "C:\\Users\\TPWODL\\New folder_Content\\StudentPerformancePrediction-End-To-End\\data\\raw\\student.csv"
pipeline = TrainingPipeline(dataset_path=dataset_path)
pipeline.run_pipeline()