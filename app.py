from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Model path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "placement_model.pkl")

# Load model
model_data = joblib.load(MODEL_PATH)

model = model_data["model"]
scaler = model_data["scaler"]

# Features actually used by the trained model
FEATURES = [
    "10th_Percentage",
    "12th_Percentage",
    "UG_CGPA",
    "Attendance",
    "Study_Hours",
    "Backlogs",
    "Technical_Skills",
    "Communication_Skills"
]


# =========================
# HOME PAGE
# =========================

@app.route("/")
def home():
    return render_template("index.html")


# =========================
# PREDICTION
# =========================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        values = {
            "10th_Percentage": float(request.form["10th_Percentage"]),
            "12th_Percentage": float(request.form["12th_Percentage"]),
            "UG_CGPA": float(request.form["UG_CGPA"]),
            "Attendance": float(request.form["Attendance"]),
            "Study_Hours": float(request.form["Study_Hours"]),
            "Backlogs": int(request.form["Backlogs"]),
            "Technical_Skills": int(request.form["Technical_Skills"]),
            "Communication_Skills": int(request.form["Communication_Skills"])
        }

        # Create dataframe in exact feature order
        input_df = pd.DataFrame(
            [values],
            columns=FEATURES
        )

        # Scale input
        scaled_input = scaler.transform(input_df)

        # Prediction
        prediction_value = model.predict(scaled_input)[0]

        # Confidence
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(scaled_input)[0]

            confidence = round(
                float(max(probabilities)) * 100,
                2
            )

        else:
            confidence = 0

        # Convert prediction
        if int(prediction_value) == 1:
            result = "PLACED"
        else:
            result = "NOT PLACED"

        return render_template(
            "index.html",
            prediction=result,
            confidence=confidence,
            show_result=True
        )

    except Exception as e:

        print("Prediction Error:", e)

        return render_template(
            "index.html",
            error=str(e),
            show_result=True
        )


# =========================
# RUN APPLICATION
# =========================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
