import numpy as np
import pandas as pd

from src.validate import DataValidator, validate_data_quality


class TestDataValidator:

    def test_validator_initialization(self):
        """Test validator can be initialized"""
        validator = DataValidator()
        assert validator is not None

    def test_schema_validation_success(self, sample_data):
        """Test schema validation passes with correct data"""
        validator = DataValidator()
        result = validator.validate_schema(sample_data)
        assert result["is_valid"]
        assert result["missing_columns"] == []

    def test_schema_validation_missing_columns(self):
        """Test schema validation fails with missing columns"""
        df = pd.DataFrame({"tenure": [1, 2], "MonthlyCharges": [50, 60]})
        validator = DataValidator()

        result = validator.validate_schema(df)
        assert not result["is_valid"]
        assert len(result["missing_columns"]) > 0

    def test_data_types_validation(self, sample_data):
        """Test data types are correct"""
        # Validator initialization removed (unused)

        # tenure should be numeric
        assert pd.api.types.is_numeric_dtype(sample_data["tenure"])
        # gender should be object/string
        assert pd.api.types.is_object_dtype(sample_data["gender"])

    def test_missing_values_detection(self):
        """Test detection of missing values"""
        df = pd.DataFrame({"tenure": [1, 2, np.nan, 4], "MonthlyCharges": [50, np.nan, 70, 80]})
        validator = DataValidator()

        missing_report = validator.check_missing_values(df)
        assert missing_report["tenure"] == 1
        assert missing_report["MonthlyCharges"] == 1

    def test_outlier_detection(self, sample_data):
        """Test outlier detection in numerical columns"""
        validator = DataValidator()
        outliers = validator.detect_outliers(sample_data, "MonthlyCharges")

        assert isinstance(outliers, pd.Series)
        assert outliers.dtype == bool

    def test_categorical_values_validation(self, sample_data):
        """Test categorical values are within expected range"""
        # Validator initialization removed (unused)

        valid_genders = {"Male", "Female"}
        assert set(sample_data["gender"].unique()).issubset(valid_genders)

    def test_data_quality_metrics(self, sample_data):
        """Test calculation of data quality metrics"""
        metrics = validate_data_quality(sample_data)

        assert "completeness" in metrics
        assert "validity" in metrics
        assert 0 <= metrics["completeness"] <= 1
        assert 0 <= metrics["validity"] <= 1


class TestDataDrift:

    def test_drift_detection_no_drift(self, sample_data):
        """Test drift detection when no drift present"""
        # Split data
        reference = sample_data[:50]
        current = sample_data[50:]

        validator = DataValidator()
        drift_report = validator.detect_drift(reference, current)

        assert "drift_detected" in drift_report
        assert isinstance(drift_report["drift_detected"], bool)

    def test_drift_detection_with_drift(self, sample_data):
        """Test drift detection when drift is present"""
        reference = sample_data.copy()
        current = sample_data.copy()

        # Introduce drift
        current["MonthlyCharges"] = current["MonthlyCharges"] * 2

        validator = DataValidator()
        drift_report = validator.detect_drift(reference, current)

        assert drift_report["drift_detected"]
