import os
from src.pipelines.prediction_pipeline import PredictionPipeline



# Correct dataset path
# Candidate paths
path1 = r"C:\Users\LENOVO\MachineLearningProhects\StudentPerformancePrediction-End-To-End\data\raw\student.csv"
path2 = r"C:\Users\TPWODL\New folder_Content\StudentPerformancePrediction-End-To-End\data\raw\student.csv"

# Condition check
if os.path.exists(path1):
    dataset_path = path1
elif os.path.exists(path2):
    dataset_path = path2
else:
    raise FileNotFoundError("❌ student.csv not found in either path!")

pipeline = PredictionPipeline(dataset_path=dataset_path)
pipeline.run_prediction_pipeline()





