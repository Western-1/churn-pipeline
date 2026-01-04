import time
import logging
from functools import wraps
from typing import Callable, Any, Dict, Optional
from prometheus_client import Summary, Gauge, Counter, Histogram, Info, generate_latest

# Setup logger
logger = logging.getLogger(__name__)

class ModelMonitor:
    def __init__(self):
        # --- 1. Metrics for Inference (API) ---
        
        # Latency (Summary is lighter, Histogram gives percentiles)
        self.prediction_latency = Summary(
            'churn_prediction_latency_seconds',
            'Time spent processing prediction'
        )
        
        # Total predictions counter with labels
        self.prediction_counter = Counter(
            'churn_predictions_total',
            'Total number of churn predictions made',
            ['prediction', 'model_version']
        )
        
        # Error counter
        self.error_counter = Counter(
            'churn_prediction_errors_total',
            'Total number of prediction errors',
            ['error_type']
        )

        # API Request details (from your snippet)
        self.api_requests_total = Counter(
            'churn_api_requests_total',
            'Total API requests',
            ['endpoint', 'method', 'status']
        )

        self.api_request_duration = Histogram(
            'churn_api_request_duration_seconds',
            'API request duration',
            ['endpoint', 'method']
        )

        # --- 2. Metrics for Validation (validate.py) ---
        self.data_quality_score = Gauge(
            'churn_data_quality_score', 
            'Data Quality Score (0-1)'
        )
        
        self.data_drift_detected = Gauge(
            'churn_data_drift_detected', 
            'Data Drift Detected (1=Yes, 0=No)'
        )

        # --- 3. Metrics for Training (train.py) ---
        self.model_accuracy = Gauge(
            'churn_model_accuracy', 
            'Current model accuracy on validation set'
        )
        
        self.model_auc_roc = Gauge(
            'churn_model_auc_roc', 
            'Current model AUC-ROC score'
        )
        
        self.model_info = Info(
            'churn_model_info',
            'Information about the current model'
        )

    # --- Methods called by validate.py ---
    def update_data_quality(self, quality_score: float, drift_detected: bool):
        """Updates data quality metrics"""
        self.data_quality_score.set(quality_score)
        self.data_drift_detected.set(1 if drift_detected else 0)

    # --- Methods called by train.py ---
    def update_model_metrics(self, accuracy: float, auc_roc: float, version: str = "latest"):
        """Updates model training metrics"""
        self.model_accuracy.set(accuracy)
        self.model_auc_roc.set(auc_roc)
        self.model_info.info({'version': str(version)})

    # --- Methods called by inference.py ---
    def observe_latency(self, seconds: float):
        self.prediction_latency.observe(seconds)

    def record_prediction(self, prediction: Any, model_version: str = "unknown"):
        self.prediction_counter.labels(
            prediction=str(prediction),
            model_version=str(model_version)
        ).inc()

    def record_error(self, error_type: str):
        self.error_counter.labels(error_type=error_type).inc()

    def get_latest_metrics(self):
        return generate_latest()


# --- Global Instance ---
# This instance is imported by other modules
monitor = ModelMonitor()


# --- Decorators ---

def track_prediction_time(func: Callable) -> Callable:
    """Decorator to track prediction latency and errors (Used in inference.py)"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Record latency
            monitor.observe_latency(duration)
            
            # Try to record prediction result if it's a dict
            if isinstance(result, dict) and 'churn_prediction' in result:
                monitor.record_prediction(
                    prediction=result['churn_prediction'],
                    model_version=result.get('model_version', 'unknown')
                )
            
            return result
        except Exception as e:
            monitor.record_error(type(e).__name__)
            raise e
    return wrapper

def track_api_request(endpoint: str, method: str):
    """Decorator for FastAPI endpoints"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                status = "error"
                raise e
            finally:
                duration = time.time() - start_time
                monitor.api_request_duration.labels(
                    endpoint=endpoint,
                    method=method
                ).observe(duration)
                
                monitor.api_requests_total.labels(
                    endpoint=endpoint,
                    method=method,
                    status=status
                ).inc()
        return wrapper
    return decorator