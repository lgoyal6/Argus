import { useState } from "react";
import RunSelector from "./components/RunSelector";
import MetricsChart from "./components/MetricsChart";
import DecisionFeed from "./components/DecisionFeed";
import { isDemoMode } from "./api";
import "./App.css";

export default function App() {
  const [selectedRun, setSelectedRun] = useState(null);

  const isLive = selectedRun?.status === "running";

  return (
    <div className="app-shell" style={styles.root}>

      {/* header */}
      <div className="app-header" style={styles.header}>
        <div style={styles.headerLeft}>
          <span style={styles.logo}>Argus</span>
          <span style={styles.tagline}>autonomous ml training debugger</span>
        </div>
        <RunSelector
          selectedRun={selectedRun}
          onSelect={setSelectedRun}
        />
      </div>

      {isDemoMode && (
        <aside className="demo-notice" aria-label="Public demo scope">
          <div>
            <strong>Bundled demo data</strong>
            <span>21 training samples, 2 detector findings, read-only</span>
          </div>
          <p>
            This page replays the repository fixture through Argus&apos;s real detector.
            No model API, backend, Supabase, or repair action runs here.
          </p>
          <a
            href="https://github.com/lgoyal6/Argus/blob/main/examples/sample_metrics.jsonl"
            target="_blank"
            rel="noreferrer"
          >
            Inspect the source fixture
          </a>
        </aside>
      )}

      {/* main content */}
      <div className="app-body" style={styles.body}>

        {/* left: metrics */}
        <div style={styles.left}>
          <MetricsChart runId={selectedRun?.id} isLive={isLive} demoMode={isDemoMode} />
        </div>

        {/* right: agent decisions */}
        <div style={styles.right}>
          <DecisionFeed runId={selectedRun?.id} isLive={isLive} demoMode={isDemoMode} />
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
    color: "#8c8c8c",
  },
  body: {
    alignItems: "start",
  },
  left: {
    minWidth: 0,
  },
  right: {
    minWidth: 0,
  },
};
