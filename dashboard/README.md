# Argus dashboard

The React front end for [Argus](../README.md). It polls the backend every 5 seconds and
plots training metrics live alongside the agent's decisions.

Built with Vite, React and Recharts.

## Components

- `RunSelector` picks which training run to watch.
- `MetricsChart` plots loss and gradient-norm series as they arrive.
- `DecisionFeed` streams the anomalies the agent flagged and the config patches it applied.

## Running it

```bash
npm install
npm run dev          # http://localhost:5173
```

The API base URL comes from `VITE_API_URL` and falls back to `http://localhost:8000`.
Start the backend first, or bring the whole stack up with `docker compose up` from the
repository root.
