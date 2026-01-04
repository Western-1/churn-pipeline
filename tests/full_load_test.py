import requests
import random
import time

url_predict = "http://localhost:8000/predict"
url_feedback = "http://localhost:8000/feedback"

def generate_data(with_drift=False, force_error=False):
    if force_error:
        # Відправляємо пустий об'єкт, щоб викликати помилку всередині функції predict
        return {} 
    
    # Решта вашого коду генерації даних...
    data = {
        "gender": random.choice(["Male", "Female"]),
        "SeniorCitizen": random.choice([0, 1]),
        "Partner": random.choice(["Yes", "No"]),
        "Dependents": random.choice(["Yes", "No"]),
        "tenure": random.randint(1, 72) if not with_drift else random.randint(200, 500),
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
        "PaymentMethod": random.choice(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]),
        "MonthlyCharges": round(random.uniform(18.0, 118.0), 2),
        "TotalCharges": round(random.uniform(18.0, 8000.0), 2)
    }
    return data

print("🚀 Starting STRESS test with real Internal Server Errors (500)...")

while True:
    make_drift = random.random() < 0.1 
    # Робимо 10% шансу на справжню помилку сервера
    make_internal_error = random.random() < 0.1 
    
    data = generate_data(with_drift=make_drift, force_error=make_internal_error)
    
    try:
        res = requests.post(url_predict, json=data)
        
        if res.status_code == 200:
            print(f"✅ Predict Success")
            # Тут відправка фідбеку...
        elif res.status_code == 500:
            print(f"🔥 Status: 500 | INTERNAL ERROR (Captured by Prometheus!)")
        elif res.status_code == 422:
            print(f"⚠️ Status: 422 | Validation error (FastAPI level)")
            
    except Exception as e:
        print(f"🚨 Connection error: {e}")
    
    time.sleep(0.1)