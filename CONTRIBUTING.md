# Contributing to Argus

Argus lets an LLM edit hyperparameters and restart a training run with nobody watching,
so the question in any PR is not "does it work" but "what can it now do that it could
not do before".

## The contract you must not break

**Every autonomous action is bounded and recorded.** The agent's entire capability
surface is the four tools declared in `agent/tools.py`: `read_config`, `read_metrics`,
`patch_config`, `rerun_training`. There is no shell, no arbitrary file write, no network
call outside the rerun. That list is the blast radius, and growing it is a design
decision, not an implementation detail.

The bounds that hold it together, all of which are easy to break by accident:

- `MAX_TOOL_ROUNDS = 10` in `agent/loop.py` caps a single invocation. Every path out of
  `run_agent()`, including the round-limit path, calls `log_decision()`. An early return
  that skips it produces an action nobody can audit afterwards.
- `logger.log_decision()` writes local JSONL *before* Supabase, and swallows Supabase
  errors. That order matters: a database outage must never lose the record of what the
  agent did to someone's training run.
- `COOLDOWN_STEPS = 500`, tracked per anomaly type in the main loop, is what stops the
  agent from patching the config every ten seconds while a spike decays.
- `patch_config` has no value bounds, so the only thing between the agent and
  `learning_rate: 1e9` is prompt rules 2 and 5 in `agent/prompts.py`. Treat that file as
  code: changing a numbered rule changes behaviour.

## Getting oriented

| Path | What lives there |
|---|---|
| `agent/detector.py` | The SPC detectors. Pure functions over a list of metric dicts, thresholds at the top of the file. |
| `agent/loop.py` | Polling loop, cooldown bookkeeping, the tool-use orchestration. |
| `agent/tools.py` | Tool schemas sent to the API, and their implementations. |
| `agent/prompts.py` | The twelve-rule system prompt and the user-prompt builder. |
| `agent/logger.py` | Dual decision logging: local JSONL then Supabase. |
| `backend/routes/` | The eight endpoints over runs, metrics and decisions. |
| `training_job/` | SmallCNN, the training loop, `config.yaml`, and `inject_anomaly.py`. |
| `dashboard/src/` | React and Recharts. `api.js` is the only place the backend URL appears. |
| `scripts/` | `demo.sh` injects all four anomaly types; `reset_demo.sh` wipes and re-creates a run. |

## Running it

```bash
cp .env.example .env          # then fill in Supabase URL/key and ANTHROPIC_API_KEY
docker compose up --build     # dashboard :5173, backend :8000, training :8001
bash scripts/reset_demo.sh    # wipes metrics, resets config, creates a fresh run
bash scripts/demo.sh          # injects all four anomaly types, 30s apart
```

The detector runs standalone, which is the fastest way to iterate on it:

```bash
python3 agent/detector.py training_job/metrics/metrics.jsonl
cd dashboard && npm install && npm run lint
```

There are no automated tests and no CI here, so exercise your change against a real run
before opening a PR and say in the description what you saw.

## What makes a good PR here

- One concern per PR.
- A new tool, or a widening of an existing one, needs a paragraph in the PR on what the
  agent can now reach and why that is acceptable with no human in the loop.
- Detector changes should come with the metrics that triggered them. `inject_anomaly.py`
  can reproduce any of the four modes on demand, so paste the JSONL rather than
  describing it.
- Thresholds live as named constants at the top of `detector.py` and `loop.py`. Keep new
  ones there rather than inline, and keep the README's detector table in step.
- Never commit `.env`, or a real Supabase project reference in `.env.example`.

## Good first areas

- **`RUN_ID` is a hardcoded UUID literal** at the top of `agent/loop.py`, and the README
  tells you to edit the source file and restart the container. `logger.set_run_id()`
  already exists and compose already passes `.env` into the agent, so reading it from the
  environment is a small, contained fix.
- **`rerun_training()` ignores its `training_dir` argument** and POSTs to a hardcoded
  `http://training:8001/rerun`, which only resolves inside the compose network. The agent
  cannot rerun training when run outside Docker.
- **The agent ignores the paths recorded on the run.** `RunCreate` carries `config_path`,
  `metrics_file` and `training_dir`, but `loop.py` never fetches the run and uses its own
  module-level constants instead, so the two can silently disagree.
- **`scripts/demo.sh` hardcodes the container name `autodebug-training-1`.** Compose
  derives that from the project directory, so the script only works if the checkout is
  named `autodebug`. `docker compose exec training` would work anywhere.
- **`agent/detector.py` has no tests** and is the easiest thing here to test: four pure
  functions over a list of dicts, with every threshold already named.

## Conduct

Be decent. Disagree about the code, not about the person.
