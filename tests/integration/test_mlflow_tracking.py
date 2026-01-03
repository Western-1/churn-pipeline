import pytest
import mlflow
from src.train import log_experiment_to_mlflow

class TestMLflowIntegration:
    
    @pytest.fixture(autouse=True)
    def setup_mlflow(self):
        """Setup MLflow for testing"""
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("test_churn_prediction")
        yield
        # Cleanup after tests
    
    def test_mlflow_connection(self):
        """Test connection to MLflow server"""
        try:
            experiments = mlflow.search_experiments()
            assert experiments is not None
        except Exception as e:
            pytest.skip(f"MLflow server not available: {e}")
    
    def test_log_params_to_mlflow(self):
        """Test logging parameters to MLflow"""
        with mlflow.start_run():
            params = {
                'max_depth': 5,
                'learning_rate': 0.1,
                'n_estimators': 100
            }
            mlflow.log_params(params)
            
            run = mlflow.active_run()
            assert run is not None
    
    def test_log_metrics_to_mlflow(self):
        """Test logging metrics to MLflow"""
        with mlflow.start_run():
            metrics = {
                'accuracy': 0.85,
                'roc_auc': 0.88,
                'precision': 0.82
            }
            mlflow.log_metrics(metrics)
            
            run = mlflow.active_run()
            assert run is not None
    
    def test_log_model_to_mlflow(self, sample_data):
        """Test logging model artifact to MLflow"""
        from sklearn.ensemble import RandomForestClassifier
        
        X = sample_data.drop(['Churn', 'customerID'], axis=1)
        y = sample_data['Churn'].map({'Yes': 1, 'No': 0})
        
        from sklearn.preprocessing import LabelEncoder
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = LabelEncoder().fit_transform(X[col])
        
        model = RandomForestClassifier(n_estimators=10)
        model.fit(X, y)
        
        with mlflow.start_run():
            mlflow.sklearn.log_model(model, "model")
            run_id = mlflow.active_run().info.run_id
            
            # Try to load model
            loaded_model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
            assert loaded_model is not None