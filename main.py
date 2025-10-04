import os
import webbrowser
from flask import Flask, render_template, request, jsonify
from src.pipelines.deployment_pipeline import DeploymentPipeline
from src.pipelines.prediction_pipeline import StudentDataInput
from src.logging.logger import get_logger
from src.exceptions.exception import CustomException
from src.constants.constants import *

logger = get_logger(__name__)
app = Flask(__name__)

# Initialize dataset path
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

# Initialize deployment pipeline (but don't run it yet)
deployment_pipeline = DeploymentPipeline(dataset_path=dataset_path)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        student = StudentDataInput(
            gender=data["gender"],
            race_ethnicity=data["race_ethnicity"],
            parental_level_of_education=data["parental_level_of_education"],
            lunch=data["lunch"],
            test_preparation_course=data["test_preparation_course"],
            reading_score=int(data["reading_score"]),
            writing_score=int(data["writing_score"])
        )

        # Make prediction (training already done at startup)
        predicted_score = deployment_pipeline.make_prediction(student)
        return jsonify({"success": True, "predicted_math_score": round(predicted_score, 2)})

    except Exception as e:
        logger.error("❌ Error during prediction", exc_info=True)
        return jsonify({"success": False, "error": str(e)})


if __name__ == "__main__":
    # Run the deployment pipeline only once at startup
    if deployment_pipeline.model_deployment_artifact is None:
        logger.info("🚀 Running Deployment Pipeline at startup...")
        deployment_pipeline.run_deployment_pipeline()
        logger.info("✅ Deployment Pipeline ready.")


    # Open browser only once
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        webbrowser.open("http://127.0.0.1:5000/")

    # Run Flask app without reloader to prevent double execution
    app.run(debug=True, use_reloader=False)

