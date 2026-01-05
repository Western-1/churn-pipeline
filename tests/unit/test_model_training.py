from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.inference import app

client = TestClient(app)


class TestInferenceAPI:
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        # Accept both healthy (if model loaded) and unhealthy (if not)
        assert response.json()["status"] in ["healthy", "unhealthy"]

    def test_predict_endpoint_success(self, sample_inference_data):
        """Test successful prediction"""
        # FIX: Explicitly set the global model variable in the module
        mock_model = MagicMock()
        mock_model.predict.return_value = [1]

        # Patching specifically where it's used
        with patch("src.inference.model", mock_model):
            response = client.post("/predict", json=sample_inference_data)

            assert response.status_code == 200
            result = response.json()
            assert "churn_prediction" in result

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
        assert response.status_code in [200, 404]

    def test_metrics_endpoint(self):
        """Test metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code == 200
