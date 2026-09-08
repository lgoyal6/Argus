import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
export const isDemoMode = import.meta.env.VITE_DEMO_MODE === "true";

const api = axios.create({
  baseURL: BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

let demoDataPromise;

const getDemoData = () => {
  if (!demoDataPromise) {
    demoDataPromise = fetch(`${import.meta.env.BASE_URL}demo-data.json`).then((response) => {
      if (!response.ok) throw new Error("could not load bundled demo data");
      return response.json();
    });
  }
  return demoDataPromise;
};

const demoResponse = async (select) => ({ data: select(await getDemoData()) });

// ── runs ───────────────────────────────────────────────────────────────────────
export const getRuns = () =>
  isDemoMode ? demoResponse((fixture) => fixture.runs) : api.get("/runs/");
export const getRun = (runId) =>
  isDemoMode
    ? demoResponse((fixture) => fixture.runs.find((run) => run.id === runId))
    : api.get(`/runs/${runId}`);
export const createRun = (body) => api.post("/runs/", body);
export const updateRunStatus = (runId, status) =>
  api.patch(`/runs/${runId}/status`, null, { params: { status } });

// ── metrics ────────────────────────────────────────────────────────────────────
export const getMetrics = (runId) =>
  isDemoMode
    ? demoResponse((fixture) => fixture.metrics[runId] || [])
    : api.get(`/runs/${runId}/metrics`);
export const syncMetrics = (runId) => api.post(`/runs/${runId}/metrics/sync`);

// ── decisions ──────────────────────────────────────────────────────────────────
export const getDecisions = (runId) =>
  isDemoMode
    ? demoResponse((fixture) => fixture.decisions[runId] || [])
    : api.get(`/runs/${runId}/decisions`);
