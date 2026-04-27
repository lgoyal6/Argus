#!/bin/bash
# Run this script while all containers are up (docker compose up) and the dashboard is open.

echo "Injecting loss_spike..."
docker exec -w /app autodebug-training-1 python3 /app/inject_anomaly.py loss_spike
sleep 30

echo "Injecting grad_explosion..."
docker exec -w /app autodebug-training-1 python3 /app/inject_anomaly.py grad_explosion
sleep 30

echo "Injecting overfitting..."
docker exec -w /app autodebug-training-1 python3 /app/inject_anomaly.py overfitting
sleep 30

echo "Injecting val_plateau..."
docker exec -w /app autodebug-training-1 python3 /app/inject_anomaly.py val_plateau
sleep 30

echo "Demo complete."
