from fastapi.testclient import TestClient

from src.inference import app

client = TestClient(app)


class TestInferenceAPI:

    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()
        assert response.json()["status"] == "healthy"

    def test_predict_endpoint_success(self, sample_inference_data):
        """Test successful prediction"""
        response = client.post("/predict", json=sample_inference_data)

        assert response.status_code == 200
        data = response.json()
        assert "churn_prediction" in data
        assert "probability" in data
        assert "message" in data
        assert data["churn_prediction"] in [0, 1]
        assert 0 <= data["probability"] <= 1

    def test_predict_endpoint_invalid_data(self):
        """Test prediction with invalid data"""
        invalid_data = {
            "gender": "Female",
            "tenure": -5,  # Invalid: negative
        }

        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error

    def test_predict_endpoint_missing_fields(self):
        """Test prediction with missing required fields"""
        incomplete_data = {"gender": "Female", "tenure": 12}

        response = client.post("/predict", json=incomplete_data)
        assert response.status_code == 422

    def test_model_info_endpoint(self):
        """Test model info endpoint"""
        response = client.get("/model/info")

        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data
        assert "model_version" in data

    def test_metrics_endpoint(self):
        """Test Prometheus metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code == 200
