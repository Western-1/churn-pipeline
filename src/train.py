import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

load_dotenv()

# --- Config ---
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "churn-prediction-exp"

def main():
    # 1. Init MLflow
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"Tracking URI: {TRACKING_URI}")

    # 2. Path Resolution (Docker vs Local)
    docker_data_path = "/opt/airflow/data/raw/churn.csv"
    local_data_path = "data/raw/churn.csv"
    
    if os.path.exists(docker_data_path):
        data_path = docker_data_path
    elif os.path.exists(local_data_path):
        data_path = local_data_path
    else:
        raise FileNotFoundError(f"Data not found in {local_data_path} or {docker_data_path}")

    print(f"Loading data: {data_path}")
    df = pd.read_csv(data_path)

    # 3. Preprocessing
    if 'customerID' in df.columns:
        df = df.drop(['customerID'], axis=1)
        
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

    # Simple encoding (Note: In prod, save encoders to artifacts!)
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Starting training...")
    
    # 4. Train & Log
    with mlflow.start_run():
        params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42
        }
        
        mlflow.log_params(params)

        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        print(f"Accuracy: {acc:.4f}")
        print(f"AUC: {auc:.4f}")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", auc)

        # Save model to MinIO
        mlflow.sklearn.log_model(model, artifact_path="model")
        print("Model saved to MLflow artifacts.")

if __name__ == "__main__":
    main()