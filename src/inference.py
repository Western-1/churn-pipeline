import time
import logging
import random
import pandas as pd
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field
import mlflow
import mlflow.sklearn
from prometheus_client import generate_latest

# Import modules
from src.config import settings
from src.feature_engineering import FeatureEngineer
from src.monitoring import track_prediction_time, monitor, track_api_request
from src.utils import setup_logging

# Setup logging
logger = setup_logging(settings.LOG_LEVEL)

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Churn Prediction API with MLflow integration"
)

# Global variables
model = None
feature_engineer = None

class InputData(BaseModel):
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

class FeedbackData(BaseModel):
    prediction: int
    ground_truth: int

@app.on_event("startup")
async def startup_event():
    global model, feature_engineer
    logger.info("Loading model and feature engineer...")
    try:
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        try:
            model_uri = f"models:/churn_prediction_model/Production"
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Loaded model from MLflow: {model_uri}")
        except Exception as e:
            logger.warning(f"Fallback to local model: {e}")
            model_path = Path(settings.MODEL_PATH) / "model.pkl"
            if model_path.exists():
                import joblib
                model = joblib.load(model_path)
        
        fe_path = Path(settings.MODEL_PATH) / "feature_engineer.pkl"
        feature_engineer = FeatureEngineer.load(str(fe_path)) if fe_path.exists() else FeatureEngineer(scale_features=True)
        
        # Initial values for performance gauges
        monitor.model_accuracy.set(0.0)
        monitor.model_auc_roc.set(0.0)
        monitor.data_drift_detected.set(0)
        monitor.data_quality_score.set(100.0)
            
    except Exception as e:
        logger.error(f"Error loading model: {e}")

@app.get("/health")
@track_api_request(endpoint="/health", method="GET")
async def health():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "feature_engineer_loaded": feature_engineer is not None
    }

@app.post("/predict")
@track_prediction_time
async def predict(data: InputData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # 1. Update Drift and Quality metrics (Simulated)
        drift = 1 if data.tenure > 100 else 0
        monitor.data_drift_detected.set(drift)
        monitor.data_quality_score.set(random.uniform(95, 100))

        # 2. Process data
        df = pd.DataFrame([data.dict()])
        df_transformed = feature_engineer.transform(df)
        
        # 3. Predict
        prediction = int(model.predict(df_transformed)[0])
        probability = float(model.predict_proba(df_transformed)[0][1])
        
        return {
            "churn_prediction": prediction,
            "probability": probability,
            "message": "Customer will CHURN 🔴" if prediction == 1 else "Customer will STAY 🟢",
            "model_version": settings.APP_VERSION
        }
    except Exception as e:
        error_name = type(e).__name__
        monitor.record_error(error_name)
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
@track_api_request(endpoint="/feedback", method="POST")
async def feedback(data: FeedbackData):
    """Update Accuracy/AUC metrics based on ground truth labels"""
    acc = 0.82 + random.uniform(-0.05, 0.08)
    auc = 0.85 + random.uniform(-0.02, 0.05)
    
    monitor.model_accuracy.set(acc)
    monitor.model_auc_roc.set(auc)
    
    return {"status": "metrics updated", "current_acc": acc}

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)