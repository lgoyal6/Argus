#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Load .env for SUPABASE_URL and SUPABASE_KEY (informational, curl uses backend)
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

echo "=== AutoDebug Demo Reset ==="

# 1. Stop all containers
echo "[1/7] Stopping containers..."
docker compose down

# 2. Clear metrics file
echo "[2/7] Clearing metrics..."
> training_job/metrics/metrics.jsonl

# 3. Reset config.yaml to sensible defaults
echo "[3/7] Resetting config.yaml..."
cat > training_job/config.yaml << 'EOF'
checkpointing:
  checkpoint_dir: checkpoints/
  save_every: 5
data:
  batch_size: 64
  dataset: cifar10
  num_workers: 2
model:
  num_classes: 10
monitoring:
  emit_every_n_steps: 10
  metrics_file: metrics/metrics.jsonl
training:
  epochs: 20
  gradient_clip: 2.0
  learning_rate: 0.0003
  optimizer: adam
  weight_decay: 0.0001
EOF

# 4. Start containers (backend needed to create run)
echo "[4/7] Starting containers..."
docker compose up -d

# 5. Wait for backend to be ready
echo "[5/7] Waiting for backend (http://localhost:8000)..."
for i in $(seq 1 30); do
  if curl -sf http://localhost:8000/runs/ > /dev/null 2>&1; then
    echo "  Backend ready."
    break
  fi
  if [ "$i" -eq 30 ]; then
    echo "  ERROR: Backend did not become ready in time."
    exit 1
  fi
  sleep 2
done

# 6. Create a new run
echo "[6/7] Creating new run in Supabase..."
RUN_RESPONSE=$(curl -sf -X POST http://localhost:8000/runs/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "demo-run-1",
    "config_path": "/app/config.yaml",
    "metrics_file": "/app/metrics/metrics.jsonl",
    "training_dir": "/app"
  }')

echo "  Run response: $RUN_RESPONSE"

RUN_ID=$(echo "$RUN_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['id'])")
echo "  New RUN_ID: $RUN_ID"

# 7. Update RUN_ID in agent/loop.py and restart
echo "[7/7] Updating RUN_ID in agent/loop.py..."
python3 - <<PYEOF
import re

path = "agent/loop.py"
content = open(path).read()
content = re.sub(r'^RUN_ID = .*', f'RUN_ID = "$RUN_ID"', content, flags=re.MULTILINE)
open(path, "w").write(content)
print(f"  Updated agent/loop.py with RUN_ID = $RUN_ID")
PYEOF

# Restart agent to pick up new RUN_ID
echo "  Restarting agent container..."
docker compose restart agent

echo ""
echo "=== Reset complete ==="
echo "  RUN_ID : $RUN_ID"
echo "  Dashboard : http://localhost:5173"
echo "  Run 'bash scripts/demo.sh' to start the demo."
