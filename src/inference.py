import os
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# 1. Ініціалізація
app = FastAPI(title="Churn Prediction Service")

# 2. Налаштування MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

# 3. Структура вхідних даних
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# 4. Допоміжна функція: Перетворення тексту в цифри
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # 1. Словник для базових значень
    mapping = {
        "Yes": 1, "No": 0,
        "No internet service": 0, "No phone service": 0,
        "Female": 0, "Male": 1
    }
    df = df.replace(mapping)
    
    # 2. Для всіх інших текстових колонок (наприклад, PaymentMethod) робимо просте кодування
    # У продакшені тут мав би бути завантажений OneHotEncoder, але для демо це спрацює.
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category').cat.codes
            
    # 3. Гарантуємо, що все стало числами
    return df.apply(pd.to_numeric, errors='coerce').fillna(0)

# 5. Завантаження моделі
def load_latest_model(experiment_name="churn-prediction-exp"):
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found.")
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=["start_time DESC"],
        max_results=1
    )
    
    if not runs:
        raise ValueError("No finished runs found.")
    
    run_id = runs[0].info.run_id
    model_uri = f"runs:/{run_id}/model"
    print(f"📥 Loading model from Run ID: {run_id}...")
    return mlflow.pyfunc.load_model(model_uri)

model = None

@app.on_event("startup")
def startup_event():
    global model
    try:
        model = load_latest_model()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")

@app.post("/predict")
def predict(customer: CustomerData):
    global model
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Створюємо DataFrame
        data = pd.DataFrame([customer.dict()])
        
        # 🔥 Перетворюємо текст на цифри
        processed_data = preprocess_data(data)
        
        # Прогноз
        prediction = model.predict(processed_data)
        result = int(prediction[0])
        
        message = "Customer will CHURN 🔴" if result == 1 else "Customer will STAY 🟢"
        
        return {
            "churn_prediction": result,
            "message": message
        }
    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}