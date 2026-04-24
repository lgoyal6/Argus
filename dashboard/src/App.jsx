import { useState } from "react";
import RunSelector from "./components/RunSelector";
import MetricsChart from "./components/MetricsChart";
import DecisionFeed from "./components/DecisionFeed";

export default function App() {
  const [selectedRunId, setSelectedRunId] = useState(null);

  return (
    <div style={styles.root}>

      {/* header */}
      <div style={styles.header}>
        <div style={styles.headerLeft}>
          <span style={styles.logo}>AutoDebug</span>
          <span style={styles.tagline}>autonomous ml training debugger</span>
        </div>
        <RunSelector
          selectedRunId={selectedRunId}
          onSelect={setSelectedRunId}
        />
      </div>

      {/* main content */}
      <div style={styles.body}>

        {/* left: metrics */}
        <div style={styles.left}>
          <MetricsChart runId={selectedRunId} />
        </div>

        {/* right: agent decisions */}
        <div style={styles.right}>
          <DecisionFeed runId={selectedRunId} />
        </div>

      </div>

    </div>
  );
}

const styles = {
  root: {
    minHeight: "100vh",
    backgroundColor: "#0a0a0a",
    color: "#fff",
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    padding: "24px",
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "24px",
    paddingBottom: "16px",
    borderBottom: "1px solid #1e1e1e",
  },
  headerLeft: {
    display: "flex",
    alignItems: "baseline",
    gap: "12px",
  },
  logo: {
    fontSize: "20px",
    fontWeight: 600,
    color: "#fff",
    letterSpacing: "-0.02em",
  },
  tagline: {
    fontSize: "13px",
    color: "#555",
  },
  body: {
    display: "grid",
    gridTemplateColumns: "1fr 420px",
    gap: "20px",
    alignItems: "start",
  },
  left: {
    minWidth: 0,
  },
  right: {
    minWidth: 0,
  },
};
