from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# =========================
# LOAD MODEL
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "placement_model.pkl")

model_data = joblib.load(MODEL_PATH)

model = model_data["model"]
scaler = model_data["scaler"]

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

print("Model loaded successfully!")
print("Model features:", FEATURES)


# =========================
# HOME
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

        data = {
            "10th_Percentage": float(request.form["10th_Percentage"]),
            "12th_Percentage": float(request.form["12th_Percentage"]),
            "UG_CGPA": float(request.form["UG_CGPA"]),
            "Attendance": float(request.form["Attendance"]),
            "Study_Hours": float(request.form["Study_Hours"]),
            "Backlogs": float(request.form["Backlogs"]),
            "Technical_Skills": float(request.form["Technical_Skills"]),
            "Communication_Skills": float(request.form["Communication_Skills"])
        }

        input_data = pd.DataFrame(
            [[
                data["10th_Percentage"],
                data["12th_Percentage"],
                data["UG_CGPA"],
                data["Attendance"],
                data["Study_Hours"],
                data["Backlogs"],
                data["Technical_Skills"],
                data["Communication_Skills"]
            ]],
            columns=FEATURES
        )

        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_scaled)[0]
            confidence = max(probability) * 100
        else:
            confidence = 100

        if prediction == 1:
            result = "PLACED"
        else:
            result = "NOT PLACED"

        return render_template(
            "index.html",
            prediction=result,
            confidence=round(confidence, 2),
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
# RUN
# =========================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )