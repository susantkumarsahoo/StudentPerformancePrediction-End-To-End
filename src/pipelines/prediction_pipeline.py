user_input = pd.DataFrame([{
    "gender": "female",
    "race_ethnicity": "group B",
    "parental_level_of_education": "bachelor's degree",
    "lunch": "standard",
    "test_preparation_course": "none",
    "reading_score": 72,
    "writing_score": 74
}])
# Load both
preprocessor = joblib.load("preprocessor.pkl")
model = joblib.load("lasso_model.pkl")

# Transform user input
X_user = preprocessor.transform(user_input)

# Predict
predicted_score = model.predict(X_user)
print("🎯 Predicted Math Score:", predicted_score[0])




