import { useEffect, useState } from "react";
import { getRuns } from "../api";

export default function RunSelector({ selectedRun, onSelect }) {
  const [runs, setRuns] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchRuns();
    const interval = setInterval(fetchRuns, 10000);
    return () => clearInterval(interval);
  }, []);

  const fetchRuns = async () => {
    try {
      const res = await getRuns();
      const data = res.data;
      setRuns(data);
      setError(null);
      setLoading(false);
    } catch (e) {
      setError("could not load runs");
      setLoading(false);
    }
  };

  // Auto-select the first (most recent) run on initial load
  useEffect(() => {
    if (runs.length > 0 && !selectedRun) {
      onSelect(runs[0]);
    }
  }, [runs]);

  // Keep selectedRun in sync when its status changes via polling
  useEffect(() => {
    if (!selectedRun || runs.length === 0) return;
    const updated = runs.find((r) => r.id === selectedRun.id);
    if (updated && updated.status !== selectedRun.status) {
      onSelect(updated);
    }
  }, [runs]);

  const statusColor = (status) => {
    if (status === "running") return "#22c55e";
    if (status === "completed") return "#3b82f6";
    if (status === "failed") return "#ef4444";
    return "#888";
  };

  if (loading) return <div style={styles.container}>loading runs...</div>;
  if (error) return <div style={styles.container}>{error}</div>;
  if (runs.length === 0) return <div style={styles.container}>no runs yet</div>;

  const isRunning = selectedRun?.status === "running";

  return (
    <div style={styles.container}>
      <label style={styles.label}>training run</label>
      <select
        style={styles.select}
        value={selectedRun?.id || ""}
        onChange={(e) => onSelect(runs.find((r) => r.id === e.target.value))}
      >
        <option value="" disabled>select a run</option>
        {runs.map((run) => (
          <option key={run.id} value={run.id}>
            {run.name} — {run.status}
          </option>
        ))}
      </select>
      {selectedRun && (
        <div style={styles.badgeRow}>
          {isRunning && (
            <span style={styles.liveDot} title="live — polling every 5s" />
          )}
          <span
            style={{
              ...styles.badge,
              backgroundColor: statusColor(selectedRun.status),
            }}
          >
            {selectedRun.status}
          </span>
        </div>
      )}
    </div>
  );
}

const styles = {
  container: {
    display: "flex",
    alignItems: "center",
    gap: "12px",
    padding: "12px 0",
  },
  label: {
    fontSize: "13px",
    color: "#888",
    fontWeight: 500,
  },
  select: {
    padding: "6px 12px",
    borderRadius: "6px",
    border: "1px solid #333",
    backgroundColor: "#1a1a1a",
    color: "#fff",
    fontSize: "14px",
    cursor: "pointer",
  },
  badgeRow: {
    display: "flex",
    alignItems: "center",
    gap: "6px",
  },
  liveDot: {
    display: "inline-block",
    width: "8px",
    height: "8px",
    borderRadius: "50%",
    backgroundColor: "#22c55e",
    animation: "pulse-dot 1.5s ease-in-out infinite",
  },
  badge: {
    padding: "3px 10px",
    borderRadius: "999px",
    fontSize: "12px",
    color: "#fff",
    fontWeight: 500,
  },
};
