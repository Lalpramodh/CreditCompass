from flask import Flask, render_template, request
import json
import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "bin", "xgboostModel.pkl")
SCHEMA_PATH = os.path.join(BASE_DIR, "data", "columns_set.json")

app = Flask(__name__, static_folder="static", template_folder="template")

with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    FEATURE_COLUMNS = list(json.load(f)["data_columns"].keys())

# Load the model once when the service starts instead of on every request.
model = joblib.load(MODEL_PATH)


def build_features(form):
    """Convert the HTML form values into the exact features used by the model."""
    values = {column: 0.0 for column in FEATURE_COLUMNS}

    numeric_fields = {
        "ApplicantIncome": form.get("applicant_income", 0),
        "CoapplicantIncome": form.get("coapplicant_income", 0),
        "LoanAmount": form.get("loan_amount", 0),
        "Loan_Amount_Term": form.get("loan_term", 0),
    }

    for column, value in numeric_fields.items():
        values[column] = float(value or 0)

    values["Gender_Male"] = float(form.get("gender", 0))
    values["Married_Yes"] = float(form.get("marital_status", 0))
    values["Education_Not Graduate"] = float(form.get("education", 0))
    values["Self_Employed_Yes"] = float(form.get("self_employed", 0))
    values["Credit_History_1.0"] = float(form.get("credit_history", 0))

    dependents = form.get("dependents", "0")
    dependent_column = f"Dependents_{dependents}"
    if dependent_column in values:
        values[dependent_column] = 1.0

    property_area = form.get("property_area", "Urban")
    property_column = f"Property_Area_{property_area}"
    if property_column in values:
        values[property_column] = 1.0

    return pd.DataFrame([values], columns=FEATURE_COLUMNS, dtype=float)


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/prediction")
def predict():
    try:
        name = request.form.get("name", "Applicant").strip() or "Applicant"
        features = build_features(request.form)
        result = int(model.predict(features)[0])

        if result == 1:
            prediction = f"Dear Mr/Mrs/Ms {name}, your loan is approved!"
        else:
            prediction = f"Sorry Mr/Mrs/Ms {name}, your loan is rejected!"

        return render_template("prediction.html", prediction=prediction)

    except (ValueError, TypeError, KeyError) as exc:
        app.logger.exception("Invalid prediction input: %s", exc)
        return render_template(
            "error.html",
            error_message="Please check the entered values and try again."
        ), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return render_template(
            "error.html",
            error_message="Something went wrong while generating the prediction."
        ), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
