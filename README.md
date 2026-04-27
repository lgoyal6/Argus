# Argus

Autonomous ML training monitor. Argus watches a live training run, detects anomalies using statistical process control, and autonomously applies hyperparameter fixes via an LLM agent — no human in the loop.

---

## How it works

```
PyTorch trainer ──► metrics.jsonl ──► Agent (claude-sonnet-4-5)
                                           │
                         detects anomaly   │   reads config, patches YAML,
                         (z-score/CUSUM)   │   reruns training, verifies fix
                                           │
                                     FastAPI backend ──► Supabase
                                           │
                                    React dashboard
```

1. The **training job** emits one JSONL line per step (loss, grad norm, val accuracy).
2. The **agent** polls the file every 10s, runs the anomaly detector, and — when something fires — calls `claude-sonnet-4-5` in a tool-use loop (≤10 rounds) to diagnose, patch `config.yaml`, rerun training, and verify the fix.
3. Every decision (tools used, patch applied, outcome) is logged to **Supabase** and local JSONL.
4. The **dashboard** polls the backend every 5s and plots everything live.

---

## Anomaly detector

Uses statistical process control instead of fixed thresholds. All four modes share a 500-step per-type cooldown to suppress duplicates.

| Mode | Condition |
|---|---|
| Loss spike | z-score > 3σ over a 20-step rolling window |
| Grad explosion | grad\_norm > 10 **or** z-score > 4σ (20-step window) |
| Val plateau | CUSUM of (val\_loss − target) drops < 0.01 over 10 steps |
| Overfitting | val/train ≥ 2.0 **and** gap strictly increasing over 5 steps |

The injection harness (`inject_anomaly.py`) can simulate any mode on demand (loss 8–15×, grad\_norm 50–200).

---

## Stack

| Layer | Tech |
|---|---|
| Training | PyTorch, CIFAR-10, SmallCNN |
| Agent | Anthropic SDK, `claude-sonnet-4-5` |
| Backend | FastAPI, Supabase (PostgreSQL) |
| Dashboard | React, Recharts, Vite |
| Infra | Docker Compose |

---

## Project structure

```
argus/
├── agent/
│   ├── loop.py          # polling loop, tool-use orchestration
│   ├── detector.py      # SPC anomaly detection (z-score, CUSUM)
│   ├── tools.py         # read_config, read_metrics, patch_config, rerun_training
│   ├── prompts.py       # 12-rule system prompt
│   └── logger.py        # dual JSONL + Supabase decision logging
├── backend/
│   ├── main.py          # FastAPI app, CORS
│   ├── db.py            # Supabase client, upsert logic
│   ├── schemas.py       # Pydantic models
│   └── routes/          # runs.py, metrics.py, decisions.py (9 endpoints)
├── training_job/
│   ├── model.py         # SmallCNN
│   ├── inject_anomaly.py
│   └── config.yaml      # lr, gradient_clip, weight_decay, etc.
├── dashboard/
│   └── src/
│       ├── components/
│       │   ├── MetricsChart.jsx   # 3 Recharts time-series plots
│       │   ├── DecisionFeed.jsx   # agent decision history
│       │   └── RunSelector.jsx    # run picker
│       └── api.js
├── scripts/
│   ├── demo.sh          # injects all 4 anomaly types sequentially
│   └── reset_demo.sh    # wipes metrics, resets config, creates new Supabase run
└── docker-compose.yml
```

---

## Setup

### Prerequisites
- Docker + Docker Compose
- Supabase project (free tier works)
- Anthropic API key

### 1. Clone and configure

```bash
git clone https://github.com/lakshgoyal06-eng/argus
cd argus
cp .env.example .env
```

Edit `.env`:

```
SUPABASE_URL=https://<your-project>.supabase.co
SUPABASE_KEY=<your-anon-key>
ANTHROPIC_API_KEY=<your-key>
VITE_API_URL=http://localhost:8000
```

### 2. Create Supabase tables

Run in the Supabase SQL editor:

```sql
create table runs (
  id text primary key,
  name text,
  config_path text,
  metrics_file text,
  training_dir text,
  status text,
  created_at float,
  updated_at float
);

create table metrics (
  run_id text,
  step int,
  epoch int,
  train_loss float,
  val_loss float,
  val_acc float,
  grad_norm float,
  timestamp float,
  anomaly_injected text,
  primary key (run_id, step)
);

create table decisions (
  id text primary key,
  run_id text,
  timestamp float,
  anomaly_types jsonb,
  tools_used jsonb,
  agent_response text,
  fixed boolean,
  status text
);
```

### 3. Start

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| Dashboard | http://localhost:5173 |
| Backend API | http://localhost:8000 |
| Training (rerun endpoint) | http://localhost:8001 |

### 4. Create a run and wire the agent

```bash
curl -X POST http://localhost:8000/runs/ \
  -H "Content-Type: application/json" \
  -d '{"name":"run-1","config_path":"/app/config.yaml","metrics_file":"/app/metrics/metrics.jsonl","training_dir":"/app"}'
```

Copy the returned `id`, set `RUN_ID` in `agent/loop.py`, then restart the agent:

```bash
docker compose restart agent
```

Or use the reset script which does all of this automatically:

```bash
bash scripts/reset_demo.sh
```

---

## Demo

With all containers running and the dashboard open at `http://localhost:5173`:

```bash
bash scripts/demo.sh
```

Injects all 4 anomaly types at 30-second intervals. Watch the agent detect each one, patch `config.yaml`, and rerun training — all visible live on the dashboard.

---

## API reference

```
GET   /runs/                    list all runs
GET   /runs/{id}                get run
POST  /runs/                    create run
PATCH /runs/{id}/status         update status (running | completed | failed)

GET   /runs/{id}/metrics        fetch all metrics for a run
POST  /runs/{id}/metrics/sync   sync metrics.jsonl → Supabase

GET   /runs/{id}/decisions      fetch agent decisions for a run
POST  /runs/{id}/decisions      insert a decision

GET   /                         health check
```
