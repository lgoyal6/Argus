import { useEffect, useState } from "react";
import { getRuns } from "../api";

export default function RunSelector({ selectedRunId, onSelect }) {
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
      setRuns(res.data);
      setError(null);
    } catch (e) {
      setError("could not load runs");
    } finally {
      setLoading(false);
    }
  };

  const statusColor = (status) => {
    if (status === "running") return "#22c55e";
    if (status === "completed") return "#3b82f6";
    if (status === "failed") return "#ef4444";
    return "#888";
  };

  if (loading) return <div style={styles.container}>loading runs...</div>;
  if (error) return <div style={styles.container}>{error}</div>;
  if (runs.length === 0) return <div style={styles.container}>no runs yet</div>;

  return (
    <div style={styles.container}>
      <label style={styles.label}>training run</label>
      <select
        style={styles.select}
        value={selectedRunId || ""}
        onChange={(e) => onSelect(e.target.value)}
      >
        <option value="" disabled>select a run</option>
        {runs.map((run) => (
          <option key={run.id} value={run.id}>
            {run.name} — {run.status}
          </option>
        ))}
      </select>
      {selectedRunId && (
        <span
          style={{
            ...styles.badge,
            backgroundColor: statusColor(
              runs.find((r) => r.id === selectedRunId)?.status
            ),
          }}
        >
          {runs.find((r) => r.id === selectedRunId)?.status}
        </span>
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
  badge: {
    padding: "3px 10px",
    borderRadius: "999px",
    fontSize: "12px",
    color: "#fff",
    fontWeight: 500,
  },
};
