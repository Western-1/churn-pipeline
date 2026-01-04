from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os

class Settings(BaseSettings):
    """Application settings with validation"""
    
    # Application
    APP_NAME: str = "Churn Prediction Pipeline"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # MLflow Configuration
    MLFLOW_TRACKING_URI: str = Field(
        default="http://localhost:5000",
        env="MLFLOW_TRACKING_URI"
    )
    MLFLOW_EXPERIMENT_NAME: str = "churn_prediction"
    MLFLOW_ARTIFACT_LOCATION: Optional[str] = None
    
    # MinIO / S3 Configuration
    MINIO_ENDPOINT: str = Field(default="localhost:9000", env="MINIO_ENDPOINT")
    MINIO_ACCESS_KEY: str = Field(default="minioadmin", env="MINIO_ACCESS_KEY")
    MINIO_SECRET_KEY: str = Field(default="minioadmin", env="MINIO_SECRET_KEY")
    MINIO_BUCKET: str = "mlflow"
    MINIO_SECURE: bool = False
    
    # Database Configuration
    POSTGRES_HOST: str = Field(default="localhost", env="POSTGRES_HOST")
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = Field(default="airflow", env="POSTGRES_USER")
    POSTGRES_PASSWORD: str = Field(default="airflow", env="POSTGRES_PASSWORD")
    POSTGRES_DB: str = "airflow"
    
    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )
    
    # Model Configuration
    MODEL_NAME: str = "xgboost_churn_classifier"
    MODEL_STAGE: str = "Production"
    MODEL_PATH: str = "models/"
    
    # Training Configuration
    TRAIN_TEST_SPLIT: float = 0.2
    RANDOM_STATE: int = 42
    CV_FOLDS: int = 5
    
    # XGBoost Parameters
    XGBOOST_MAX_DEPTH: int = 6
    XGBOOST_LEARNING_RATE: float = 0.1
    XGBOOST_N_ESTIMATORS: int = 100
    XGBOOST_MIN_CHILD_WEIGHT: int = 1
    XGBOOST_SUBSAMPLE: float = 0.8
    XGBOOST_COLSAMPLE_BYTREE: float = 0.8
    
    # Data Paths
    DATA_RAW_PATH: str = "data/raw/"
    DATA_PROCESSED_PATH: str = "data/processed/"
    DATA_REPORTS_PATH: str = "data/reports/"
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    
    # Monitoring
    ENABLE_PROMETHEUS: bool = True
    PROMETHEUS_PORT: int = 9090
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

# Singleton instance
settings = Settings()