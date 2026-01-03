import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from xgboost import XGBClassifier
import joblib
import logging

# Import new modules
from src.config import settings
from src.feature_engineering import FeatureEngineer
from src.monitoring import monitor
from src.utils import setup_logging, save_json

# Setup logging
logger = setup_logging(settings.LOG_LEVEL)


def train_model(X_train, y_train, params=None):
    """
    Train XGBoost model with given parameters
    
    Args:
        X_train: Training features
        y_train: Training labels
        params: Model hyperparameters (optional)
    
    Returns:
        Trained model
    """
    logger.info("Starting model training...")
    
    # Use params from config if not provided
    if params is None:
        params = {
            'max_depth': settings.XGBOOST_MAX_DEPTH,
            'learning_rate': settings.XGBOOST_LEARNING_RATE,
            'n_estimators': settings.XGBOOST_N_ESTIMATORS,
            'min_child_weight': settings.XGBOOST_MIN_CHILD_WEIGHT,
            'subsample': settings.XGBOOST_SUBSAMPLE,
            'colsample_bytree': settings.XGBOOST_COLSAMPLE_BYTREE,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': settings.RANDOM_STATE
        }
    
    # Train model
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)
    
    logger.info("Model training complete")
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model and return metrics
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
    
    Returns:
        Dictionary of metrics
    """
    logger.info("Evaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
    }
    
    logger.info(f"Model Metrics: {metrics}")
    
    # Update monitoring metrics
    monitor.update_model_metrics(
        accuracy=metrics['accuracy'],
        auc_roc=metrics['roc_auc'],
        version="latest"
    )
    
    return metrics


def save_model(model, path):
    """
    Save model to disk
    
    Args:
        model: Model to save
        path: Path to save model
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")


def train_and_log_model(data_path=None):
    """
    Complete training pipeline with MLflow logging
    
    Args:
        data_path: Path to training data (optional)
    
    Returns:
        Dictionary with model and metrics
    """
    logger.info("Starting complete training pipeline...")
    
    # Setup MLflow
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
    
    # Load data
    if data_path is None:
        data_path = Path(settings.DATA_RAW_PATH) / "churn.csv"
    
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Separate features and target
    X = df.drop(['Churn', 'customerID'], axis=1, errors='ignore')
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=settings.TRAIN_TEST_SPLIT,
        random_state=settings.RANDOM_STATE,
        stratify=y
    )
    
    # Feature engineering
    logger.info("Applying feature engineering...")
    feature_engineer = FeatureEngineer(scale_features=True)
    X_train_transformed = feature_engineer.fit_transform(X_train)
    X_test_transformed = feature_engineer.transform(X_test)
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"training_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Log parameters
        params = {
            'max_depth': settings.XGBOOST_MAX_DEPTH,
            'learning_rate': settings.XGBOOST_LEARNING_RATE,
            'n_estimators': settings.XGBOOST_N_ESTIMATORS,
            'train_test_split': settings.TRAIN_TEST_SPLIT,
            'random_state': settings.RANDOM_STATE,
        }
        mlflow.log_params(params)
        
        # Train model
        model = train_model(X_train_transformed, y_train, params)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test_transformed, y_test)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Save feature engineer
        fe_path = Path(settings.MODEL_PATH) / "feature_engineer.pkl"
        feature_engineer.save(str(fe_path))
        mlflow.log_artifact(str(fe_path))
        
        # Save metrics to file
        metrics_path = Path(settings.MODEL_PATH) / "metrics.json"
        save_json(metrics, str(metrics_path))
        
        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow run_id: {run_id}")
        
        return {
            'model': model,
            'feature_engineer': feature_engineer,
            'metrics': metrics,
            'run_id': run_id
        }


def log_experiment_to_mlflow(model, metrics, params):
    """
    Log existing experiment to MLflow (for testing)
    
    Args:
        model: Trained model
        metrics: Dictionary of metrics
        params: Dictionary of parameters
    """
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
    
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
        
        return mlflow.active_run().info.run_id


if __name__ == "__main__":
    # Run training pipeline
    result = train_and_log_model()
    print(f"Training complete! Run ID: {result['run_id']}")
    print(f"Metrics: {result['metrics']}")