#!/bin/bash
set -e

echo "📊 Setting up monitoring stack..."

NETWORK_NAME="churn-pipeline_churn-network"

if [ -z "$(docker network ls --filter name=^${NETWORK_NAME}$ -q)" ]; then
  echo "⚠️  Network '${NETWORK_NAME}' not found!"
  echo "   Please start the main pipeline first using: docker-compose up -d"
  exit 1
else
  echo "✅ Network '${NETWORK_NAME}' found."
fi

echo "🧹 Cleaning up old monitoring containers..."
docker rm -f churn-prometheus churn-grafana churn-alertmanager churn-node-exporter 2>/dev/null || true

echo "🚀 Starting monitoring services..."
docker compose -f monitoring/docker-compose.monitoring.yml up -d

echo "⏳ Waiting for services to initialize..."
sleep 10

echo "✅ Monitoring stack setup complete!"
echo "Access points:"
echo "  Prometheus: http://localhost:9090"
echo "  Grafana:    http://localhost:3000 (admin/admin)"
echo "  Alerts:     http://localhost:9093"
