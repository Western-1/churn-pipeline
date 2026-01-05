from unittest.mock import MagicMock

from fastapi.testclient import TestClient

import src.inference
from src.inference import app

client = TestClient(app)


class TestInferenceAPI:
    def setup_method(self):
        """Setup method run before each test"""
        self.mock_model = MagicMock()
        self.mock_model.predict.return_value = [1]
        self.mock_model.predict_proba.return_value = [[0.2, 0.8]]

        src.inference.model = self.mock_model

    def teardown_method(self):
        """Cleanup after tests"""
        src.inference.model = None

    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_predict_endpoint_success(self, sample_inference_data):
        """Test successful prediction"""
        response = client.post("/predict", json=sample_inference_data)

        assert response.status_code == 200
        result = response.json()
        assert "churn_prediction" in result
        assert result["churn_prediction"] == 1

    def test_predict_endpoint_invalid_data(self):
        """Test prediction with invalid data"""
        response = client.post("/predict", json={"invalid": "data"})
        assert response.status_code == 422

    def test_predict_endpoint_missing_fields(self, sample_inference_data):
        """Test prediction with missing required fields"""
        data = sample_inference_data.copy()
        del data["Contract"]
        response = client.post("/predict", json=data)
        assert response.status_code == 422

    def test_model_info_endpoint(self):
        """Test model info endpoint"""
        response = client.get("/model/info")
        assert response.status_code == 200
        assert "model_name" in response.json()

    def test_metrics_endpoint(self):
        """Test metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code == 200
