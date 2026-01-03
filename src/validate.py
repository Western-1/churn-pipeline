import pandas as pd
import numpy as np
from pathlib import Path
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
import logging

# Import new modules
from src.config import settings
from src.monitoring import monitor
from src.utils import setup_logging

# Setup logging
logger = setup_logging(settings.LOG_LEVEL)


class DataValidator:
    """Data validation and drift detection"""
    
    def __init__(self):
        self.required_columns = [
            'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
            'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
            'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'
        ]
    
    def validate_schema(self, df):
        """
        Validate dataframe schema
        
        Args:
            df: DataFrame to validate
        
        Returns:
            Dictionary with validation results
        """
        logger.info("Validating data schema...")
        
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        
        result = {
            'is_valid': len(missing_columns) == 0,
            'missing_columns': missing_columns,
            'total_columns': len(df.columns),
            'required_columns': len(self.required_columns)
        }
        
        if not result['is_valid']:
            logger.warning(f"Schema validation failed. Missing columns: {missing_columns}")
        else:
            logger.info("Schema validation passed")
        
        return result
    
    def check_missing_values(self, df):
        """
        Check for missing values
        
        Args:
            df: DataFrame to check
        
        Returns:
            Dictionary with missing value counts
        """
        logger.info("Checking for missing values...")
        
        missing = df.isnull().sum()
        missing_dict = missing[missing > 0].to_dict()
        
        if missing_dict:
            logger.warning(f"Found missing values: {missing_dict}")
        else:
            logger.info("No missing values found")
        
        return missing_dict
    
    def detect_outliers(self, df, column, method='iqr', threshold=1.5):
        """
        Detect outliers in a numerical column
        
        Args:
            df: DataFrame
            column: Column name
            method: 'iqr' or 'zscore'
            threshold: Threshold for outlier detection
        
        Returns:
            Boolean series indicating outliers
        """
        if column not in df.columns:
            logger.warning(f"Column {column} not found")
            return pd.Series([False] * len(df))
        
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            outliers = (df[column] < (Q1 - threshold * IQR)) | (df[column] > (Q3 + threshold * IQR))
        elif method == 'zscore':
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            outliers = z_scores > threshold
        else:
            raise ValueError(f"Unknown method: {method}")
        
        outlier_count = outliers.sum()
        if outlier_count > 0:
            logger.info(f"Found {outlier_count} outliers in {column}")
        
        return outliers
    
    def detect_drift(self, reference_data, current_data):
        """
        Detect data drift using Evidently
        
        Args:
            reference_data: Reference DataFrame
            current_data: Current DataFrame
        
        Returns:
            Dictionary with drift detection results
        """
        logger.info("Detecting data drift...")
        
        # Create Evidently report
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_data, current_data=current_data)
        
        # Extract drift results
        report_dict = report.as_dict()
        
        # Get drift status
        drift_detected = report_dict['metrics'][0]['result']['dataset_drift']
        
        # Save report
        report_path = Path(settings.DATA_REPORTS_PATH) / "data_drift_report.html"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(report_path))
        
        logger.info(f"Drift report saved to {report_path}")
        
        if drift_detected:
            logger.warning("Data drift detected!")
            monitor.update_data_quality(quality_score=0.5, drift_detected=True)
        else:
            logger.info("No significant data drift detected")
            monitor.update_data_quality(quality_score=1.0, drift_detected=False)
        
        return {
            'drift_detected': drift_detected,
            'report_path': str(report_path)
        }


def validate_data_quality(df):
    """
    Calculate data quality metrics
    
    Args:
        df: DataFrame to evaluate
    
    Returns:
        Dictionary with quality metrics
    """
    logger.info("Calculating data quality metrics...")
    
    # Completeness: percentage of non-null values
    completeness = 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
    
    # Validity: percentage of values within expected ranges
    # (simplified example)
    validity = 1.0  # Assume all values are valid for now
    
    metrics = {
        'completeness': float(completeness),
        'validity': float(validity),
        'total_rows': len(df),
        'total_columns': len(df.columns)
    }
    
    logger.info(f"Quality metrics: {metrics}")
    
    # Update monitoring
    monitor.update_data_quality(
        quality_score=completeness,
        drift_detected=False
    )
    
    return metrics


def run_validation(df):
    """
    Run complete validation pipeline
    
    Args:
        df: DataFrame to validate
    
    Returns:
        Dictionary with all validation results
    """
    validator = DataValidator()
    
    results = {
        'schema': validator.validate_schema(df),
        'missing_values': validator.check_missing_values(df),
        'quality_metrics': validate_data_quality(df),
        'is_valid': True
    }
    
    # Overall validation
    results['is_valid'] = results['schema']['is_valid']
    
    return results