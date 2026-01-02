FROM apache/airflow:2.9.2-python3.10

USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Копіюємо requirements з папки docker/airflow
COPY docker/airflow/requirements.txt /requirements.txt

# Встановлюємо бібліотеки
RUN pip install --no-cache-dir -r /requirements.txt