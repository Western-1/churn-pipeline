import pandas as pd
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

BASE_DIR = "/opt/airflow" if os.path.exists("/opt/airflow") else "."
REPORT_DIR = os.path.join(BASE_DIR, "data/reports")
DATA_PATH = os.path.join(BASE_DIR, "data/raw/data.csv")

os.makedirs(REPORT_DIR, exist_ok=True)

def main():
    print("Validating data...")
    
    full_df = pd.read_csv(DATA_PATH)
    
    reference_data = full_df.iloc[:3000]
    current_data = full_df.iloc[3000:]
    
    report = Report(metrics=[
        DataDriftPreset(), 
    ])
    
    report.run(reference_data=reference_data, current_data=current_data)
    
    report_path = os.path.join(REPORT_DIR, "data_drift_report.html")
    report.save_html(report_path)
    print(f"Report saved to {report_path}")
    
    result = report.as_dict()
    drift_share = result['metrics'][0]['result']['drift_share']
    
    print(f"Drift Share: {drift_share}")
    
    if drift_share > 0.5:
        raise Exception("Data Drift detected! Stopping pipeline.")
    
    print("Data validation passed.")

if __name__ == "__main__":
    main()