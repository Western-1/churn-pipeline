from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from prometheus_client import generate_latest

# Import new modules
from src.config import settings
from src.feature_engineering import FeatureEngineer
from src.monitoring import track_prediction_time, monitor
from src.utils import setup_logging

# Setup logging
logger = setup_logging(settings.LOG_LEVEL)

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Churn Prediction API with MLflow integration"
)

# Global variables for model and feature engineer
model = None
feature_engineer = None


class InputData(BaseModel):
    """Input data schema for predictions"""
    gender: str = Field(..., description="Customer gender")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Senior citizen flag")
    Partner: str = Field(..., description="Has partner")
    Dependents: str = Field(..., description="Has dependents")
    tenure: int = Field(..., ge=0, description="Months as customer")
    PhoneService: str = Field(..., description="Has phone service")
    MultipleLines: str = Field(..., description="Has multiple lines")
    InternetService: str = Field(..., description="Internet service type")
    OnlineSecurity: str = Field(..., description="Has online security")
    OnlineBackup: str = Field(..., description="Has online backup")
    DeviceProtection: str = Field(..., description="Has device protection")
    TechSupport: str = Field(..., description="Has tech support")
    StreamingTV: str = Field(..., description="Has streaming TV")
    StreamingMovies: str = Field(..., description="Has streaming movies")
    Contract: str = Field(..., description="Contract type")
    PaperlessBilling: str = Field(..., description="Has paperless billing")
    PaymentMethod: str = Field(..., description="Payment method")
    MonthlyCharges: float = Field(..., gt=0, description="Monthly charges")
    TotalCharges: float = Field(..., ge=0, description="Total charges")


@app.on_event("startup")
async def load_model():
    """Load model and feature engineer on startup"""
    global model, feature_engineer
    
    logger.info("Loading model and feature engineer...")
    
    try:
        # Setup MLflow
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        
        # Load model from MLflow Model Registry
        try:
            model_uri = f"models:/churn_prediction_model/Production"
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Loaded model from MLflow: {model_uri}")
        except Exception as e:
            logger.warning(f"Could not load from registry: {e}")
            # Fallback to local model
            model_path = Path(settings.MODEL_PATH) / "model.pkl"
            if model_path.exists():
                import joblib
                model = joblib.load(model_path)
                logger.info(f"Loaded model from local path: {model_path}")
            else:
                logger.error("No model found!")
                model = None
        
        # Load feature engineer
        fe_path = Path(settings.MODEL_PATH) / "feature_engineer.pkl"
        if fe_path.exists():
            feature_engineer = FeatureEngineer.load(str(fe_path))
            logger.info(f"Loaded feature engineer from {fe_path}")
        else:
            logger.warning("Feature engineer not found, using default")
            feature_engineer = FeatureEngineer(scale_features=True)
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model = None
        feature_engineer = None


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Churn Prediction API",
        "version": settings.APP_VERSION,
        "status": "running"
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "feature_engineer_loaded": feature_engineer is not None
    }


@app.post("/predict")
@track_prediction_time
async def predict(data: InputData):
    """
    Make churn prediction
    
    Args:
        data: Input data for prediction
    
    Returns:
        Prediction result with probability
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        input_dict = data.dict()
        df = pd.DataFrame([input_dict])
        
        # Apply feature engineering
        if feature_engineer is not None:
            df_transformed = feature_engineer.transform(df)
        else:
            # Fallback: simple encoding
            from sklearn.preprocessing import LabelEncoder
            for col in df.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
            df_transformed = df
        
        # Make prediction
        prediction = int(model.predict(df_transformed)[0])
        probability = float(model.predict_proba(df_transformed)[0][1])
        
        result = {
            "churn_prediction": prediction,
            "probability": probability,
            "message": "Customer will CHURN 🔴" if prediction == 1 else "Customer will STAY 🟢",
            "model_version": settings.APP_VERSION
        }
        
        logger.info(f"Prediction: {prediction}, Probability: {probability:.4f}")
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/model/info")
async def model_info():
    """Get model information"""
    return {
        "model_name": settings.MODEL_NAME,
        "model_stage": settings.MODEL_STAGE,
        "model_loaded": model is not None,
        "mlflow_tracking_uri": settings.MLFLOW_TRACKING_URI,
        "version": settings.APP_VERSION
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.API_HOST,
        port=settings.API_PORT,
        log_level=settings.LOG_LEVEL.lower()
    )
