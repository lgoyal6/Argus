# AutoDebug — Build Progress

## Phase 1: Training Job ✓
- [x] model.py — SmallCNN defined (2 conv layers, dropout, 10-class classifier)
- [x] config.yaml — hyperparameters + monitoring config
- [x] train.py — training loop with metric emission every N steps
- [x] inject_anomaly.py — injects loss_spike, grad_explosion, val_plateau, overfitting
- [x] requirements.txt
- [x] verified train.py runs and emits metrics to metrics.jsonl

## Phase 2: Agent ✓
- [x] detector.py — rule-based detection for all 4 anomaly types
- [x] tools.py — read_config, read_metrics, patch_config, rerun_training
- [x] prompts.py — system prompt + user prompt builder
- [x] loop.py — main polling loop + agentic tool call orchestration
- [x] logger.py — local JSONL + Supabase write
- [x] requirements.txt
- [x] verified agent detects loss_spike and patches learning rate autonomously

## Phase 3: Backend ✓
- [x] schemas.py — Pydantic models for runs, metrics, decisions
- [x] db.py — Supabase client, CRUD for all three tables
- [x] main.py — FastAPI app, CORS, router registration
- [x] routes/runs.py — GET/POST/PATCH run endpoints
- [x] routes/metrics.py — GET metrics + sync from file
- [x] routes/decisions.py — GET/POST decisions
- [x] requirements.txt
- [x] verified GET /runs/ returns 200
- [x] verified POST /runs/ creates run in Supabase
- [x] verified metrics sync populates charts

## Phase 4: Dashboard ✓
- [x] api.js — axios wrappers for all backend endpoints
- [x] RunSelector.jsx — dropdown with live status badge
- [x] MetricsChart.jsx — loss, grad norm, val accuracy charts with anomaly reference lines
- [x] DecisionFeed.jsx — agent decision cards with anomaly tags, tools used, fixed badge
- [x] App.jsx — two-column layout, run state management
- [x] verified dashboard renders live metrics
- [x] verified run selector shows Supabase runs

## Phase 5: Docker + Deployment
- [ ] Dockerfile.training
- [ ] Dockerfile.agent
- [ ] Dockerfile.backend
- [ ] docker-compose.yml
- [ ] AWS/GCP deployment

## Milestones
- [x] first full end to end run — agent detected loss_spike, reduced lr 0.001 → 0.0005
- [x] Supabase connected and storing runs + metrics
- [x] dashboard showing live loss curves and grad norm
- [ ] agent decisions appearing in dashboard decision feed
- [ ] docker-compose up starts everything with one command
- [ ] deployed to AWS/GCP
