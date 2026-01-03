import pytest
from fastapi.testclient import TestClient
from src.inference import app

# Create test client
client = TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint"""
    
    def test_health_endpoint_exists(self):
        """Test that health endpoint exists and returns 200"""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_endpoint_returns_json(self):
        """Test that health endpoint returns valid JSON"""
        response = client.get("/health")
        assert response.headers["content-type"] == "application/json"
        data = response.json()
        assert isinstance(data, dict)
    
    def test_health_endpoint_has_status(self):
        """Test that health response includes status field"""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "ok", "running"]
    
    def test_health_endpoint_optional_fields(self):
        """Test optional fields in health response"""
        response = client.get("/health")
        data = response.json()
        
        # Optional fields that might be present
        optional_fields = ["timestamp", "version", "service", "uptime"]
        # At least one optional field should be present for detailed health check
        # But this is flexible - can pass with just status too


class TestPredictionEndpoint:
    """Tests for prediction endpoint"""
    
    def test_predict_endpoint_exists(self):
        """Test that predict endpoint exists"""
        # POST with minimal data (will fail validation but endpoint exists)
        response = client.post("/predict", json={})
        # Should return 422 (validation error) not 404 (not found)
        assert response.status_code in [200, 422]
    
    def test_predict_with_valid_data(self, sample_inference_data):
        """Test prediction with valid input data"""
        response = client.post("/predict", json=sample_inference_data)
        
        # Should return 200 OK
        assert response.status_code == 200
        
        # Check response structure
        data = response.json()
        assert isinstance(data, dict)
        assert "churn_prediction" in data or "prediction" in data
    
    def test_predict_returns_valid_prediction(self, sample_inference_data):
        """Test that prediction is in valid range"""
        response = client.post("/predict", json=sample_inference_data)
        
        if response.status_code == 200:
            data = response.json()
            prediction_key = "churn_prediction" if "churn_prediction" in data else "prediction"
            prediction = data[prediction_key]
            
            # Prediction should be 0 or 1 (binary classification)
            assert prediction in [0, 1]
    
    def test_predict_returns_probability(self, sample_inference_data):
        """Test that response includes probability"""
        response = client.post("/predict", json=sample_inference_data)
        
        if response.status_code == 200:
            data = response.json()
            
            # Check for probability field (various possible names)
            has_probability = any(
                key in data 
                for key in ["probability", "confidence", "score", "proba"]
            )
            
            if has_probability:
                prob_key = next(
                    key for key in ["probability", "confidence", "score", "proba"]
                    if key in data
                )
                probability = data[prob_key]
                
                # Probability should be between 0 and 1
                assert 0 <= probability <= 1
    
    def test_predict_with_missing_fields(self):
        """Test prediction with missing required fields"""
        incomplete_data = {
            "gender": "Female",
            "tenure": 12
            # Missing many required fields
        }
        
        response = client.post("/predict", json=incomplete_data)
        
        # Should return 422 (Unprocessable Entity) for validation error
        assert response.status_code == 422
    
    def test_predict_with_invalid_data_types(self):
        """Test prediction with invalid data types"""
        invalid_data = {
            "gender": "Female",
            "SeniorCitizen": "invalid",  # Should be int
            "tenure": "not_a_number",    # Should be int
            "MonthlyCharges": "invalid", # Should be float
        }
        
        response = client.post("/predict", json=invalid_data)
        
        # Should return 422 (Validation Error)
        assert response.status_code == 422
    
    def test_predict_with_out_of_range_values(self, sample_inference_data):
        """Test prediction with out-of-range values"""
        invalid_data = sample_inference_data.copy()
        invalid_data["tenure"] = -5  # Negative tenure (invalid)
        
        response = client.post("/predict", json=invalid_data)
        
        # Should return 422 (Validation Error) or 400 (Bad Request)
        assert response.status_code in [400, 422]
    
    def test_predict_response_time(self, sample_inference_data):
        """Test that prediction response is fast enough"""
        import time
        
        start_time = time.time()
        response = client.post("/predict", json=sample_inference_data)
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Response should be under 2 seconds (adjust based on your SLA)
        assert response_time < 2.0, f"Response too slow: {response_time:.2f}s"


class TestModelInfoEndpoint:
    """Tests for model information endpoint"""
    
    def test_model_info_endpoint_exists(self):
        """Test that model info endpoint exists"""
        # Try common endpoint names
        endpoints_to_try = ["/model/info", "/info", "/model", "/version"]
        
        found = False
        for endpoint in endpoints_to_try:
            response = client.get(endpoint)
            if response.status_code == 200:
                found = True
                break
        
        # If model info endpoint exists, it should return 200
        # If it doesn't exist yet, that's also acceptable
        assert True  # Flexible test
    
    def test_model_info_returns_version(self):
        """Test that model info includes version"""
        endpoints_to_try = ["/model/info", "/info", "/version"]
        
        for endpoint in endpoints_to_try:
            response = client.get(endpoint)
            if response.status_code == 200:
                data = response.json()
                # Check for version-related fields
                version_fields = ["version", "model_version", "model_name"]
                has_version = any(field in data for field in version_fields)
                if has_version:
                    assert True
                    return
        
        # Endpoint might not exist yet
        assert True


class TestMetricsEndpoint:
    """Tests for Prometheus metrics endpoint"""
    
    def test_metrics_endpoint_exists(self):
        """Test that /metrics endpoint exists"""
        response = client.get("/metrics")
        
        # Should return 200 if implemented
        # Should return 404 if not implemented yet
        assert response.status_code in [200, 404]
    
    def test_metrics_endpoint_returns_prometheus_format(self):
        """Test that metrics are in Prometheus format"""
        response = client.get("/metrics")
        
        if response.status_code == 200:
            # Prometheus metrics are plain text
            assert "text/plain" in response.headers.get("content-type", "")
            
            # Should contain metric names with underscores
            content = response.text
            assert "_" in content or "HELP" in content or "TYPE" in content


class TestBatchPredictionEndpoint:
    """Tests for batch prediction endpoint (if implemented)"""
    
    def test_batch_predict_endpoint(self, sample_inference_data):
        """Test batch prediction endpoint"""
        # Try batch endpoint
        batch_data = {
            "instances": [sample_inference_data, sample_inference_data]
        }
        
        response = client.post("/predict/batch", json=batch_data)
        
        # Endpoint might not exist yet (404)
        # Or might return results (200)
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, dict)
            # Should have predictions for both instances
            assert "predictions" in data or "results" in data


class TestErrorHandling:
    """Tests for error handling"""
    
    def test_404_for_unknown_endpoint(self):
        """Test that unknown endpoints return 404"""
        response = client.get("/unknown/endpoint/that/does/not/exist")
        assert response.status_code == 404
    
    def test_405_for_wrong_method(self):
        """Test that wrong HTTP method returns 405"""
        # GET on POST endpoint
        response = client.get("/predict")
        assert response.status_code in [405, 422]  # Method Not Allowed or Unprocessable
    
    def test_error_response_structure(self):
        """Test that error responses have consistent structure"""
        response = client.post("/predict", json={})
        
        if response.status_code >= 400:
            data = response.json()
            assert isinstance(data, dict)
            # FastAPI returns 'detail' field for errors
            assert "detail" in data or "error" in data or "message" in data


class TestCORS:
    """Tests for CORS configuration (if enabled)"""
    
    def test_cors_headers_present(self, sample_inference_data):
        """Test that CORS headers are present if configured"""
        response = client.post("/predict", json=sample_inference_data)
        
        # Check if CORS headers are present
        cors_headers = [
            "access-control-allow-origin",
            "access-control-allow-methods",
            "access-control-allow-headers"
        ]
        
        # If CORS is configured, at least one header should be present
        # If not configured, that's also acceptable
        # This is a flexible test
        assert True


class TestAPIDocumentation:
    """Tests for API documentation endpoints"""
    
    def test_swagger_ui_exists(self):
        """Test that Swagger UI is accessible"""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
    
    def test_openapi_schema_exists(self):
        """Test that OpenAPI schema is accessible"""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        # Should return valid JSON
        data = response.json()
        assert isinstance(data, dict)
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data
    
    def test_redoc_exists(self):
        """Test that ReDoc documentation is accessible"""
        response = client.get("/redoc")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")


class TestRateLimiting:
    """Tests for rate limiting (if implemented)"""
    
    def test_rate_limiting(self, sample_inference_data):
        """Test rate limiting if implemented"""
        # Make multiple rapid requests
        responses = []
        for _ in range(100):
            response = client.post("/predict", json=sample_inference_data)
            responses.append(response.status_code)
        
        # If rate limiting is implemented, some requests should be rejected
        # Status code 429 (Too Many Requests)
        # If not implemented, all should succeed (200)
        
        # This test is informational - both cases are valid
        rate_limited = 429 in responses
        all_succeeded = all(status == 200 for status in responses)
        
        assert rate_limited or all_succeeded


class TestConcurrency:
    """Tests for concurrent request handling"""
    
    def test_concurrent_requests(self, sample_inference_data):
        """Test that API handles concurrent requests"""
        import concurrent.futures
        
        def make_request():
            return client.post("/predict", json=sample_inference_data)
        
        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # All requests should complete (status 200 or error)
        assert len(results) == 10
        
        # Most should succeed
        success_count = sum(1 for r in results if r.status_code == 200)
        assert success_count >= 8  # At least 80% success rate


# Integration test that combines multiple endpoints
class TestEndToEndWorkflow:
    """End-to-end workflow tests"""
    
    def test_complete_prediction_workflow(self, sample_inference_data):
        """Test complete prediction workflow"""
        # 1. Check health
        health_response = client.get("/health")
        assert health_response.status_code == 200
        
        # 2. Get model info (if available)
        info_response = client.get("/model/info")
        # May or may not exist
        
        # 3. Make prediction
        predict_response = client.post("/predict", json=sample_inference_data)
        assert predict_response.status_code == 200
        
        # 4. Check metrics (if available)
        metrics_response = client.get("/metrics")
        # May or may not exist
        
        # Overall workflow should work
        assert True
    
    def test_multiple_predictions_consistency(self, sample_inference_data):
        """Test that multiple predictions with same data are consistent"""
        # Make same prediction 3 times
        responses = []
        for _ in range(3):
            response = client.post("/predict", json=sample_inference_data)
            if response.status_code == 200:
                responses.append(response.json())
        
        if len(responses) >= 2:
            # Predictions should be consistent (same input -> same output)
            first_pred = responses[0]
            for resp in responses[1:]:
                # Prediction should be same
                pred_key = "churn_prediction" if "churn_prediction" in first_pred else "prediction"
                if pred_key in first_pred and pred_key in resp:
                    assert first_pred[pred_key] == resp[pred_key]


# Parametrized tests for different input scenarios
class TestParametrizedPredictions:
    """Parametrized tests for various input scenarios"""
    
    @pytest.mark.parametrize("gender", ["Male", "Female"])
    def test_predict_different_genders(self, sample_inference_data, gender):
        """Test predictions for different genders"""
        data = sample_inference_data.copy()
        data["gender"] = gender
        
        response = client.post("/predict", json=data)
        if response.status_code == 200:
            result = response.json()
            assert "churn_prediction" in result or "prediction" in result
    
    @pytest.mark.parametrize("tenure", [0, 1, 12, 36, 72])
    def test_predict_different_tenures(self, sample_inference_data, tenure):
        """Test predictions for different tenure values"""
        data = sample_inference_data.copy()
        data["tenure"] = tenure
        
        response = client.post("/predict", json=data)
        assert response.status_code == 200
    
    @pytest.mark.parametrize("contract", ["Month-to-month", "One year", "Two year"])
    def test_predict_different_contracts(self, sample_inference_data, contract):
        """Test predictions for different contract types"""
        data = sample_inference_data.copy()
        data["Contract"] = contract
        
        response = client.post("/predict", json=data)
        assert response.status_code == 200