import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv
import os

load_dotenv()

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("churn-prediction-exp")

def main():
    print(f"Tracking URI set to: {tracking_uri}")
    print("Loading data...")
    
    docker_data_path = "/opt/airflow/data/raw/data.csv"
    local_data_path = "data/raw/data.csv"
    
    if os.path.exists(docker_data_path):
        data_path = docker_data_path
    elif os.path.exists(local_data_path):
        data_path = local_data_path
    else:
        raise FileNotFoundError("Could not find data.csv neither in /opt/airflow/data nor in local folder.")

    print(f"Reading data from: {data_path}")
    df = pd.read_csv(data_path)

    if 'customerID' in df.columns:
        df = df.drop(['customerID'], axis=1)
        
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Starting training...")
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

        # Логуємо модель
        mlflow.sklearn.log_model(model, artifact_path="model")
        print("Model saved to MLflow.")

if __name__ == "__main__":
    main()