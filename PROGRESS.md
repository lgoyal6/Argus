# AutoDebug — Build Progress

## Phase 1: Training Job ✓
- [x] model.py — SmallCNN defined
- [x] config.yaml — hyperparameters + monitoring config
- [x] train.py — training loop with metric emission
- [x] inject_anomaly.py — anomaly injection for testing
- [x] requirements.txt — dependencies
- [x] verified train.py runs and emits metrics

## Phase 2: Agent ✓
- [x] detector.py — rule-based anomaly detection
- [x] tools.py — read_config, read_metrics, patch_config, rerun_training
- [x] prompts.py — system prompt + user prompt builder
- [x] loop.py — main polling + agentic tool call loop
- [x] logger.py — local JSONL logging, Supabase hook ready
- [x] requirements.txt

## Phase 3: Backend ✓
- [x] schemas.py
- [x] db.py
- [x] main.py
- [x] routes/runs.py
- [x] routes/metrics.py
- [x] routes/decisions.py
- [x] requirements.txt

## Phase 4: Dashboard
- [ ] MetricsChart.jsx
- [ ] DecisionFeed.jsx
- [ ] RunSelector.jsx
- [ ] App.jsx
- [ ] api.js

## Phase 5: Docker + Deployment
- [ ] Dockerfiles
- [ ] docker-compose.yml
- [ ] AWS deployment
