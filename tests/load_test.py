import random
import time

import requests

url_predict = "http://localhost:8000/predict"
url_feedback = "http://localhost:8000/feedback"


def generate_data(with_drift=False):
    data = {
        "gender": random.choice(["Male", "Female"]),
        "SeniorCitizen": random.choice([0, 1]),
        "Partner": random.choice(["Yes", "No"]),
        "Dependents": random.choice(["Yes", "No"]),
        "tenure": (
            random.randint(1, 72) if not with_drift else random.randint(200, 500)
        ),
        "PhoneService": random.choice(["Yes", "No"]),
        "MultipleLines": random.choice(["No phone service", "No", "Yes"]),
        "InternetService": random.choice(["DSL", "Fiber optic", "No"]),
        "OnlineSecurity": random.choice(["No internet service", "No", "Yes"]),
        "OnlineBackup": random.choice(["No internet service", "No", "Yes"]),
        "DeviceProtection": random.choice(["No internet service", "No", "Yes"]),
        "TechSupport": random.choice(["No internet service", "No", "Yes"]),
        "StreamingTV": random.choice(["No internet service", "No", "Yes"]),
        "StreamingMovies": random.choice(["No internet service", "No", "Yes"]),
        "Contract": random.choice(["Month-to-month", "One year", "Two year"]),
        "PaperlessBilling": random.choice(["Yes", "No"]),
        "PaymentMethod": random.choice(
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ]
        ),
        "MonthlyCharges": round(random.uniform(18.0, 118.0), 2),
        "TotalCharges": round(random.uniform(18.0, 8000.0), 2),
    }
    return data


print("🚀 Starting ENHANCED load test...")

while True:
    make_drift = random.random() < 0.1 
    data = generate_data(with_drift=make_drift)

    try:
        res = requests.post(url_predict, json=data)
        if res.status_code == 200:
            pred_data = res.json()
            prediction = pred_data["churn_prediction"]

            is_correct = random.random() > 0.15
            ground_truth = prediction if is_correct else (1 - prediction)

            feedback = {"prediction": prediction, "ground_truth": int(ground_truth)}
            requests.post(url_feedback, json=feedback)

            print(f"✅ Predict: {prediction} | Actual: {ground_truth} | Drift: {make_drift}")

    except Exception as e:
        print(f"🚨 Error: {e}")

    time.sleep(0.2)
