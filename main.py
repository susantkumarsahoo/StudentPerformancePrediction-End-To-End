import sys
import traceback
from flask import Flask, request, jsonify, render_template
from src.exceptions.exception import CustomException
from src.logging.logger import get_logger
from src.entity.artifact_entity import ModelDeploymentArtifact
from src.models.predictor import ModelPredictor
from src.pipelines.prediction_pipeline import StudentDataInput

logger = get_logger(__name__)
app = Flask(__name__)

# -------------------------------
# Load artifacts at API startup
# -------------------------------
try:
    deployment_artifact = ModelDeploymentArtifact(
        deployment_model_path="artifacts/model.pkl",
        deployment_preprocessor_path="artifacts/preprocessor.pkl"
    )

    predictor = ModelPredictor(deployment_artifact)
    logger.info("Model and preprocessor successfully loaded at API startup.")

except Exception as e:
    logger.error(f"Error loading artifacts: {str(e)}")
    raise CustomException(e, sys)


# -------------------------------
# Home route (for testing / UI)
# -------------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")   # optional HTML form


# -------------------------------
# Prediction API
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()  # JSON input from user
        logger.info(f"Received input JSON: {data}")

        # Map request JSON → StudentDataInput
        student_data = StudentDataInput(
            gender=data["gender"],
            race_ethnicity=data["race_ethnicity"],
            parental_education=data["parental_level_of_education"],
            lunch=data["lunch"],
            test_prep_course=data["test_preparation_course"],
            reading_score=int(data["reading_score"]),
            writing_score=int(data["writing_score"])
        )

        # Convert to DataFrame
        input_df = student_data.to_dataframe()

        # Run prediction
        prediction = predictor.predict(input_df)

        response = {"predicted_math_score": prediction}
        return jsonify(response)

    except Exception as e:
        logger.error(f"Prediction error: {traceback.format_exc()}")
        raise CustomException(e, sys)


# -------------------------------
# Run Flask app
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

