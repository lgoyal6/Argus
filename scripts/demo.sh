#!/bin/bash
# Run this script while all containers are up (docker compose up) and the dashboard is open.
#
# The injections go through `docker compose exec` rather than `docker exec` on a
# literal container name. Compose names containers `<project>-training-1`, and the
# project defaults to the directory the checkout sits in, so a hardcoded name is only
# right for whoever wrote it: this script used to say `autodebug-training-1` and failed
# with "No such container" for every clone made after the project was renamed to Argus.
#
# `set -e` matters for the same reason. Without it each injection failed, the script
# printed "Demo complete." anyway, and the only symptom was a dashboard that never
# changed.
set -euo pipefail

inject() {
    echo "Injecting $1..."
    docker compose exec -T -w /app training python3 /app/inject_anomaly.py "$1"
}

inject loss_spike
sleep 30

inject grad_explosion
sleep 30

inject overfitting
sleep 30

inject val_plateau
sleep 30

echo "Demo complete."
