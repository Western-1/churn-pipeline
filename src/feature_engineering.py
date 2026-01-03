import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Dict, List, Optional
import joblib
from pathlib import Path


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom feature engineering transformer
    Handles encoding, scaling, and feature creation
    """
    
    def __init__(self, scale_features: bool = True):
        self.scale_features = scale_features
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[StandardScaler] = None
        self.feature_names_: Optional[List[str]] = None
        self.categorical_columns_: List[str] = []
        self.numerical_columns_: List[str] = []
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit encoders and scaler on training data"""
        X = X.copy()
        
        # Identify column types
        self.categorical_columns_ = X.select_dtypes(include=['object']).columns.tolist()
        self.numerical_columns_ = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Fit label encoders for categorical columns
        for col in self.categorical_columns_:
            le = LabelEncoder()
            # Handle missing values
            X[col] = X[col].fillna('missing')
            le.fit(X[col])
            self.label_encoders[col] = le
        
        # Create engineered features for fitting scaler
        X_transformed = self._create_features(X)
        
        # Fit scaler on numerical features
        if self.scale_features:
            self.scaler = StandardScaler()
            numerical_features = X_transformed.select_dtypes(include=[np.number]).columns
            self.scaler.fit(X_transformed[numerical_features])
        
        self.feature_names_ = X_transformed.columns.tolist()
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted encoders and scaler"""
        X = X.copy()
        
        # Encode categorical columns
        for col in self.categorical_columns_:
            if col in X.columns:
                X[col] = X[col].fillna('missing')
                # Handle unseen categories
                X[col] = X[col].apply(
                    lambda x: x if x in self.label_encoders[col].classes_ 
                    else self.label_encoders[col].classes_[0]
                )
                X[col] = self.label_encoders[col].transform(X[col])
        
        # Create engineered features
        X = self._create_features(X)
        
        # Scale numerical features
        if self.scale_features and self.scaler is not None:
            numerical_features = X.select_dtypes(include=[np.number]).columns
            X[numerical_features] = self.scaler.transform(X[numerical_features])
        
        return X
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X)
    
    def _create_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features"""
        X = X.copy()
        
        # Feature 1: Charges per month (avoid division by zero)
        if 'TotalCharges' in X.columns and 'tenure' in X.columns:
            X['charges_per_month'] = X['TotalCharges'] / (X['tenure'] + 1)
        
        # Feature 2: Tenure groups
        if 'tenure' in X.columns:
            X['tenure_group'] = pd.cut(
                X['tenure'],
                bins=[0, 12, 24, 48, 72],
                labels=['0-1yr', '1-2yr', '2-4yr', '4-6yr']
            )
            # Encode tenure_group if it was created
            if 'tenure_group' in X.columns:
                X['tenure_group'] = X['tenure_group'].astype(str)
                if 'tenure_group' not in self.label_encoders:
                    le = LabelEncoder()
                    le.fit(X['tenure_group'])
                    self.label_encoders['tenure_group'] = le
                X['tenure_group'] = self.label_encoders['tenure_group'].transform(
                    X['tenure_group']
                )
        
        # Feature 3: Service combination
        if all(col in X.columns for col in ['PhoneService', 'InternetService']):
            X['has_multiple_services'] = (
                (X['PhoneService'] == 'Yes') & 
                (X['InternetService'] != 'No')
            ).astype(int)
        
        # Feature 4: Payment reliability indicator
        if 'Contract' in X.columns:
            X['long_term_contract'] = (
                X['Contract'].isin(['One year', 'Two year'])
            ).astype(int)
        
        return X
    
    def save(self, path: str):
        """Save feature engineer to disk"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
    
    @staticmethod
    def load(path: str) -> 'FeatureEngineer':
        """Load feature engineer from disk"""
        return joblib.load(path)