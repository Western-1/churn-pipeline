import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv  # <--- 1. Спочатку імпортуємо
import os

load_dotenv()  # <--- 2. Потім викликаємо (читає .env файл)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("churn-prediction-exp")

def main():
    print("Loading data...")
    # Читаємо дані
    df = pd.read_csv("data/raw/data.csv")

    # Препроцесинг
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

        # Використовуємо sklearn flavor для XGBClassifier
        mlflow.sklearn.log_model(model, artifact_path="model")
        print("Model saved to MLflow.")

if __name__ == "__main__":
    main()