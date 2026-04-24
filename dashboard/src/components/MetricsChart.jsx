import { useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";
import { getMetrics } from "../api";

export default function MetricsChart({ runId }) {
  const [metrics, setMetrics] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!runId) return;
    fetchMetrics();
    const interval = setInterval(fetchMetrics, 5000);
    return () => clearInterval(interval);
  }, [runId]);

  const fetchMetrics = async () => {
    try {
      const res = await getMetrics(runId);
      setMetrics(res.data);
      setError(null);
    } catch (e) {
      setError("could not load metrics");
    } finally {
      setLoading(false);
    }
  };

  // find steps where anomalies were injected
  const anomalySteps = metrics
    .filter((m) => m.anomaly_injected)
    .map((m) => m.step);

  if (!runId) return <div style={styles.empty}>select a run to view metrics</div>;
  if (loading) return <div style={styles.empty}>loading metrics...</div>;
  if (error) return <div style={styles.empty}>{error}</div>;
  if (metrics.length === 0) return <div style={styles.empty}>no metrics yet</div>;

  return (
    <div style={styles.container}>
      <h2 style={styles.title}>training metrics</h2>

      {/* loss chart */}
      <div style={styles.chartWrap}>
        <p style={styles.chartLabel}>loss</p>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={metrics} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a2a2a" />
            <XAxis
              dataKey="step"
              stroke="#555"
              tick={{ fill: "#888", fontSize: 11 }}
              label={{ value: "step", position: "insideBottom", offset: -2, fill: "#555", fontSize: 11 }}
            />
            <YAxis stroke="#555" tick={{ fill: "#888", fontSize: 11 }} />
            <Tooltip
              contentStyle={{ backgroundColor: "#1a1a1a", border: "1px solid #333", borderRadius: "6px" }}
              labelStyle={{ color: "#888" }}
              itemStyle={{ color: "#fff" }}
            />
            <Legend wrapperStyle={{ fontSize: "12px", color: "#888" }} />
            {anomalySteps.map((step) => (
              <ReferenceLine key={step} x={step} stroke="#ef4444" strokeDasharray="4 4" />
            ))}
            <Line type="monotone" dataKey="train_loss" stroke="#3b82f6" dot={false} strokeWidth={1.5} />
            <Line type="monotone" dataKey="val_loss" stroke="#f59e0b" dot={false} strokeWidth={1.5} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* grad norm chart */}
      <div style={styles.chartWrap}>
        <p style={styles.chartLabel}>gradient norm</p>
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={metrics} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a2a2a" />
            <XAxis
              dataKey="step"
              stroke="#555"
              tick={{ fill: "#888", fontSize: 11 }}
            />
            <YAxis stroke="#555" tick={{ fill: "#888", fontSize: 11 }} />
            <Tooltip
              contentStyle={{ backgroundColor: "#1a1a1a", border: "1px solid #333", borderRadius: "6px" }}
              labelStyle={{ color: "#888" }}
              itemStyle={{ color: "#fff" }}
            />
            {anomalySteps.map((step) => (
              <ReferenceLine key={step} x={step} stroke="#ef4444" strokeDasharray="4 4" />
            ))}
            <Line type="monotone" dataKey="grad_norm" stroke="#22c55e" dot={false} strokeWidth={1.5} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* val accuracy chart */}
      <div style={styles.chartWrap}>
        <p style={styles.chartLabel}>validation accuracy</p>
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={metrics} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a2a2a" />
            <XAxis
              dataKey="step"
              stroke="#555"
              tick={{ fill: "#888", fontSize: 11 }}
            />
            <YAxis stroke="#555" tick={{ fill: "#888", fontSize: 11 }} domain={[0, 1]} />
            <Tooltip
              contentStyle={{ backgroundColor: "#1a1a1a", border: "1px solid #333", borderRadius: "6px" }}
              labelStyle={{ color: "#888" }}
              itemStyle={{ color: "#fff" }}
            />
            {anomalySteps.map((step) => (
              <ReferenceLine key={step} x={step} stroke="#ef4444" strokeDasharray="4 4" />
            ))}
            <Line type="monotone" dataKey="val_acc" stroke="#a855f7" dot={false} strokeWidth={1.5} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

const styles = {
  container: {
    backgroundColor: "#111",
    borderRadius: "10px",
    padding: "20px",
    border: "1px solid #222",
  },
  title: {
    fontSize: "15px",
    fontWeight: 500,
    color: "#fff",
    marginBottom: "20px",
  },
  chartWrap: {
    marginBottom: "24px",
  },
  chartLabel: {
    fontSize: "12px",
    color: "#888",
    marginBottom: "8px",
    textTransform: "uppercase",
    letterSpacing: "0.05em",
  },
  empty: {
    color: "#555",
    fontSize: "14px",
    padding: "40px 0",
    textAlign: "center",
  },
};
