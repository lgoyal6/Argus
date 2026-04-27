#!/bin/bash
# Run this script while all containers are up (docker compose up) and the dashboard is open.

echo "Injecting loss_spike..."
docker exec autodebug-training-1 python3 inject_anomaly.py loss_spike
sleep 60

echo "Injecting grad_explosion..."
docker exec autodebug-training-1 python3 inject_anomaly.py grad_explosion
sleep 60

echo "Injecting overfitting..."
docker exec autodebug-training-1 python3 inject_anomaly.py overfitting
sleep 60

echo "Injecting val_plateau..."
docker exec autodebug-training-1 python3 inject_anomaly.py val_plateau
sleep 60

echo "Demo complete."
