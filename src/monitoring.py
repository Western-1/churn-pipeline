from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest
from functools import wraps
import time
from typing import Callable, Any
import logging

logger = logging.getLogger(__name__)

# Define Prometheus metrics

# Prediction metrics
prediction_counter = Counter(
    'churn_predictions_total',
    'Total number of churn predictions made',
    ['prediction', 'model_version']
)

prediction_latency = Histogram(
    'churn_prediction_latency_seconds',
    'Time spent processing prediction request',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

prediction_errors = Counter(
    'churn_prediction_errors_total',
    'Total number of prediction errors',
    ['error_type']
)

# Model metrics
model_version_info = Info(
    'churn_model',
    'Information about the current model'
)

model_accuracy = Gauge(
    'churn_model_accuracy',
    'Current model accuracy on validation set'
)

model_auc = Gauge(
    'churn_model_auc_roc',
    'Current model AUC-ROC score'
)

# Data quality metrics
data_drift_detected = Gauge(
    'churn_data_drift_detected',
    'Whether data drift has been detected (1) or not (0)'
)

data_quality_score = Gauge(
    'churn_data_quality_score',
    'Overall data quality score (0-1)'
)

# API metrics
api_requests_total = Counter(
    'churn_api_requests_total',
    'Total API requests',
    ['endpoint', 'method', 'status']
)

api_request_duration = Histogram(
    'churn_api_request_duration_seconds',
    'API request duration',
    ['endpoint', 'method']
)


def track_prediction_time(func: Callable) -> Callable:
    """Decorator to track prediction latency"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            prediction_latency.observe(duration)
            
            # Track prediction result
            if isinstance(result, dict) and 'churn_prediction' in result:
                model_version = result.get('model_version', 'unknown')
                prediction_counter.labels(
                    prediction=str(result['churn_prediction']),
                    model_version=model_version
                ).inc()
            
            return result
        
        except Exception as e:
            prediction_errors.labels(error_type=type(e).__name__).inc()
            raise
    
    return wrapper


def track_api_request(endpoint: str, method: str):
    """Decorator to track API requests"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            try:
                result = await func(*args, **kwargs)
                return result
            
            except Exception as e:
                status = "error"
                raise
            
            finally:
                duration = time.time() - start_time
                api_request_duration.labels(
                    endpoint=endpoint,
                    method=method
                ).observe(duration)
                
                api_requests_total.labels(
                    endpoint=endpoint,
                    method=method,
                    status=status
                ).inc()
        
        return wrapper
    return decorator


class ModelMonitor:
    """Monitor model performance and data quality"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def update_model_metrics(
        self,
        accuracy: float,
        auc_roc: float,
        version: str
    ):
        """Update model performance metrics"""
        model_accuracy.set(accuracy)
        model_auc.set(auc_roc)
        model_version_info.info({
            'version': version,
            'algorithm': 'xgboost'
        })
        
        self.logger.info(
            f"Updated model metrics: accuracy={accuracy:.4f}, "
            f"auc_roc={auc_roc:.4f}, version={version}"
        )
    
    def update_data_quality(
        self,
        quality_score: float,
        drift_detected: bool
    ):
        """Update data quality metrics"""
        data_quality_score.set(quality_score)
        data_drift_detected.set(1 if drift_detected else 0)
        
        if drift_detected:
            self.logger.warning(
                f"Data drift detected! Quality score: {quality_score:.4f}"
            )
    
    def get_metrics(self) -> bytes:
        """Get all metrics in Prometheus format"""
        return generate_latest()