import axios from "axios";
const BASE_URL = "http://3.137.141.220:8000";
const api = axios.create({
  baseURL: BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

// ── runs ───────────────────────────────────────────────────────────────────────
export const getRuns = () => api.get("/runs/");
export const getRun = (runId) => api.get(`/runs/${runId}`);
export const createRun = (body) => api.post("/runs/", body);
export const updateRunStatus = (runId, status) =>
  api.patch(`/runs/${runId}/status`, null, { params: { status } });

// ── metrics ────────────────────────────────────────────────────────────────────
export const getMetrics = (runId) => api.get(`/runs/${runId}/metrics`);
export const syncMetrics = (runId) => api.post(`/runs/${runId}/metrics/sync`);

// ── decisions ──────────────────────────────────────────────────────────────────
export const getDecisions = (runId) => api.get(`/runs/${runId}/decisions`);
