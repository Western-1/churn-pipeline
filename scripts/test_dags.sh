!/bin/bash
set -e

echo "🧪 Testing Airflow DAGs..."

# Test DAG syntax
echo "Checking DAG syntax..."
docker compose exec airflow-webserver python -m airflow dags list

# Test training pipeline
echo "Testing training pipeline..."
docker compose exec airflow-webserver \
  python -m airflow dags test churn_training_pipeline 2026-01-03

# Test monitoring DAG
echo "Testing monitoring DAG..."
docker compose exec airflow-webserver \
  python -m airflow dags test churn_monitoring 2026-01-03

echo "✅ All DAGs tested successfully!"
EOF

chmod +x scripts/test_dags.sh
