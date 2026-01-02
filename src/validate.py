import pandas as pd
import os
import mlflow
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# --- Config ---
BASE_DIR = "/opt/airflow" if os.path.exists("/opt/airflow") else "."
REPORT_DIR = os.path.join(BASE_DIR, "data/reports")
DATA_PATH = os.path.join(BASE_DIR, "data/raw/churn.csv")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

os.makedirs(REPORT_DIR, exist_ok=True)

def main():
    # 1. Init MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("churn-data-validation")

    print(f"Validating data from: {DATA_PATH}")
    
    # 2. Load Data
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data not found at {DATA_PATH}. Run 'dvc pull' first.")
        
    full_df = pd.read_csv(DATA_PATH)
    
    # 3. Split: Reference (historical) vs Current (new)
    reference_data = full_df.iloc[:3000]
    current_data = full_df.iloc[3000:]
    
    # 4. Generate Drift Report
    report = Report(metrics=[
        DataDriftPreset(), 
    ])
    
    report.run(reference_data=reference_data, current_data=current_data)
    
    # 5. Save Report Locally
    report_filename = "data_drift_report.html"
    report_path = os.path.join(REPORT_DIR, report_filename)
    report.save_html(report_path)
    print(f"Report saved: {report_path}")
    
    # 6. Extract Metrics
    result = report.as_dict()
    drift_share = result['metrics'][0]['result']['drift_share']
    dataset_drift = result['metrics'][0]['result']['dataset_drift']
    
    print(f"Drift Share: {drift_share}")
    print(f"Dataset Drift: {dataset_drift}")

    # 7. Log to MLflow
    print("Logging to MLflow...")
    with mlflow.start_run(run_name="data_validation_run"):
        mlflow.log_metric("drift_share", drift_share)
        mlflow.log_param("dataset_drift_detected", dataset_drift)
        
        # Upload HTML report to MinIO artifacts
        mlflow.log_artifact(report_path)
        print("Report uploaded to MinIO.")

    # 8. Quality Gate
    if drift_share > 0.5:
        raise Exception(f"Severe Drift (share={drift_share})! Pipeline stopped.")
    
    print("Validation passed.")

if __name__ == "__main__":
    main()