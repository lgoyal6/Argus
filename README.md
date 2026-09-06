# Argus

Autonomous ML training monitor. Argus watches a live training run, detects anomalies using statistical process control, and autonomously applies hyperparameter fixes via an LLM agent - no human in the loop.

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
2. The **agent** polls the file every 10s, runs the anomaly detector, and - when something fires - calls `claude-sonnet-4-5` in a tool-use loop (≤10 rounds) to diagnose, patch `config.yaml`, rerun training, and verify the fix.
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

**There is no credential-free way to run the product.** Supabase is the only
store `backend/db.py` knows how to talk to - there is no SQLite fallback and no
demo mode - so without a project the dashboard loads and every panel on it is
empty. What that actually looks like, if you try it with the placeholder `.env`:
containers all stay up, `GET /` answers `{"status":"ok"}`, and `GET /runs/`
answers `500` with `{"detail":"internal error","error_id":"..."}`. The cause is
deliberately not in the response body; it is in `docker compose logs backend`,
as `ConnectError` on a hostname that does not resolve.

The Anthropic key buys the other half. The detector is arithmetic and needs
nothing: with a placeholder key the training container still trains, the agent
still polls, and `docker compose logs agent` still shows real detections
(`{"event": "detection.anomaly", "type": "grad_explosion", "step": 420}` on a
run left alone for four minutes). It is the *fix* that needs the key, and it
does not degrade gracefully - the first detection raises
`anthropic.AuthenticationError: 401 invalid x-api-key`, the traceback goes to
the log, and the agent container exits 1 and stays exited, because nothing in
`docker-compose.yml` restarts it.

### What you can run without either account

The test suite drives the real FastAPI app with `db.get_client` replaced by an
in-memory store and `supabase.create_client` poisoned, so nothing can reach a
project even by accident. It needs no Docker, no keys and about three seconds:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
```

432 pass and one is a deliberate xfail. They cover the API contract and its
error shaping, the z-score and CUSUM detector on synthetic series, the metrics
tailer, the agent's sandbox and recovery behaviour, and the flat-layout import
graph the container actually uses. It is the honest answer to "does this work"
for anyone who is not going to create a Supabase project to find out.

Run it from a `git clone`, not from a downloaded archive. `test_secret_lifecycle`
asks git which files are tracked and which the ignore rules cover, so outside a
checkout ten of its cases fail on `git` exiting 128 while reporting it as
".env is not ignored by .gitignore", which is not what went wrong.

### 1. Clone and configure

```bash
git clone https://github.com/lgoyal6/Argus
cd Argus
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
bash scripts/reset_demo.sh
```

That is the whole step. It wipes metrics, resets `config.yaml`, creates a run,
writes the returned id into `agent/loop.py` and restarts the agent.

The same thing by hand, which is worth reading once because it shows where the
run id actually lives:

```bash
curl -X POST http://localhost:8000/runs/ \
  -H "Content-Type: application/json" \
  -d '{"name":"run-1","config_path":"/app/config.yaml","metrics_file":"/app/metrics/metrics.jsonl","training_dir":"/app"}'
```

Copy the returned `id`, set `RUN_ID` in `agent/loop.py`, then restart the agent:

```bash
docker compose restart agent
```

`RUN_ID` is a constant in the source rather than an environment variable, so
"configuring" the agent means editing a file the repository ships and the reset
script rewrites. A clone starts with the previous run's uuid still in it, which
is why an unconfigured agent posts to `/runs/<somebody else's id>/metrics/sync`
every ten seconds and gets a 500. That is a known rough edge, not a broken
install.

---

## Demo

With all containers running and the dashboard open at `http://localhost:5173`:

```bash
bash scripts/demo.sh
```

Injects all 4 anomaly types at 30-second intervals. Watch the agent detect each one, patch `config.yaml`, and rerun training - all visible live on the dashboard.

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
