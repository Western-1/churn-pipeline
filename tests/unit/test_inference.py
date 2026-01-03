import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.train import train_model, evaluate_model, save_model
from sklearn.metrics import accuracy_score, roc_auc_score

class TestModelTraining:
    
    def test_train_model_returns_model(self, sample_data):
        """Test that training returns a model object"""
        X = sample_data.drop(['Churn', 'customerID'], axis=1)
        y = sample_data['Churn'].map({'Yes': 1, 'No': 0})
        
        # Simple preprocessing
        from sklearn.preprocessing import LabelEncoder
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = LabelEncoder().fit_transform(X[col])
        
        model = train_model(X, y, params={'max_depth': 3, 'n_estimators': 10})
        
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
    
    def test_model_predictions_valid_range(self, sample_data):
        """Test model predictions are in valid range [0, 1]"""
        X = sample_data.drop(['Churn', 'customerID'], axis=1)
        y = sample_data['Churn'].map({'Yes': 1, 'No': 0})
        
        from sklearn.preprocessing import LabelEncoder
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = LabelEncoder().fit_transform(X[col])
        
        model = train_model(X, y)
        predictions = model.predict(X)
        
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_model_evaluate_metrics(self, sample_data):
        """Test evaluation returns expected metrics"""
        X = sample_data.drop(['Churn', 'customerID'], axis=1)
        y = sample_data['Churn'].map({'Yes': 1, 'No': 0})
        
        from sklearn.preprocessing import LabelEncoder
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = LabelEncoder().fit_transform(X[col])
        
        model = train_model(X, y)
        metrics = evaluate_model(model, X, y)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'roc_auc' in metrics
        
        # Check metrics are in valid range
        for metric, value in metrics.items():
            assert 0 <= value <= 1, f"{metric} out of range: {value}"
    
    def test_model_save_load(self, sample_data, temp_data_dir):
        """Test model can be saved and loaded"""
        X = sample_data.drop(['Churn', 'customerID'], axis=1)
        y = sample_data['Churn'].map({'Yes': 1, 'No': 0})
        
        from sklearn.preprocessing import LabelEncoder
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = LabelEncoder().fit_transform(X[col])
        
        model = train_model(X, y)
        
        # Save model
        model_path = temp_data_dir / "model.pkl"
        save_model(model, model_path)
        
        assert model_path.exists()
        
        # Load and test
        import joblib
        loaded_model = joblib.load(model_path)
        
        predictions = loaded_model.predict(X[:5])
        assert len(predictions) == 5