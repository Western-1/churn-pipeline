import os
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# 1. Init App
app = FastAPI(title="Churn Prediction Service")

# 2. Config
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_URI)

# 3. Input Schema
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

# 4. Preprocessing
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Map binary values
    mapping = {
        "Yes": 1, "No": 0,
        "No internet service": 0, "No phone service": 0,
        "Female": 0, "Male": 1
    }
    df = df.replace(mapping)
    
    # Encode categorical columns
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category').cat.codes
            
    # Ensure numeric format
    return df.apply(pd.to_numeric, errors='coerce').fillna(0)

# 5. Load Model from Registry
def load_latest_model(experiment_name="churn-prediction-exp"):
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found.")
    
    # Fetch latest successful run
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

# Global model instance
model = None

@app.on_event("startup")
def startup_event():
    global model
    try:
        model = load_latest_model()
        print("✅ Model loaded!")
    except Exception as e:
        print(f"❌ Load failed: {e}")

@app.post("/predict")
def predict(customer: CustomerData):
    global model
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # 1. Prepare Data
        data = pd.DataFrame([customer.dict()])
        processed_data = preprocess_data(data)
        
        # 2. Predict
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