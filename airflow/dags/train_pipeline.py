from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    'owner': 'andriy',
    'start_date': datetime(2024, 1, 1),
    'retries': 0,
}

with DAG(
    dag_id='churn_training_pipeline',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=['mlops', 'churn']
) as dag:

    # ВАЖЛИВО: додаємо 'cd /opt/airflow && ...'
    # Це переходить в корінь проекту, щоб Python бачив папку 'data' і 'src'
    
    validate_data = BashOperator(
        task_id='validate_data',
        bash_command='cd /opt/airflow && export PYTHONPATH=$PYTHONPATH:/opt/airflow && python src/validate.py'
    )

    train_model = BashOperator(
        task_id='train_model',
        bash_command='cd /opt/airflow && export PYTHONPATH=$PYTHONPATH:/opt/airflow && python src/train.py'
    )

    validate_data >> train_model