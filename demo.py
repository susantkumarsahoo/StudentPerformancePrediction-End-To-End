from src.pipelines.training_pipeline import TrainingPipeline

if __name__ == "__main__":
    dataset_path = "C:\\Users\\TPWODL\\New folder_Content\\StudentPerformancePrediction-End-To-End\\data\\raw\\student.csv"   # put your dataset path here
    pipeline = TrainingPipeline(dataset_path=dataset_path)
    pipeline.run_pipeline()


