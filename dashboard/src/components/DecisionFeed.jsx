import { useEffect, useState } from "react";
import { getDecisions } from "../api";

const RESPONSE_PREVIEW_LENGTH = 300;

export default function DecisionFeed({ runId, isLive }) {
  const [decisions, setDecisions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expanded, setExpanded] = useState(new Set());

  useEffect(() => {
    if (!runId) return;
    setLoading(true);
    fetchDecisions();

    // Only keep polling while the run is live
    if (!isLive) return;
    const interval = setInterval(fetchDecisions, 5000);
    return () => clearInterval(interval);
  }, [runId, isLive]);

  const fetchDecisions = async () => {
    try {
      const res = await getDecisions(runId);
      setDecisions(res.data);
      setError(null);
    } catch (e) {
      setError("could not load decisions");
    } finally {
      setLoading(false);
    }
  };

  const toggleExpanded = (id) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };

  const fixedBadge = (fixed) => {
    if (fixed === true) return { label: "fixed", color: "#22c55e" };
    if (fixed === false) return { label: "failed", color: "#ef4444" };
    return { label: "uncertain", color: "#888" };
  };

  const formatTime = (ts) => new Date(ts * 1000).toLocaleTimeString();

  if (!runId) return <div style={styles.empty}>select a run to view agent decisions</div>;
  if (loading) return <div style={styles.empty}>loading decisions...</div>;
  if (error) return <div style={styles.empty}>{error}</div>;
  if (decisions.length === 0) return <div style={styles.empty}>no agent decisions yet</div>;

  return (
    <div style={styles.container}>
      <h2 style={styles.title}>agent decisions</h2>
      {decisions.map((d) => {
        const badge = fixedBadge(d.fixed);
        const isExpanded = expanded.has(d.id);
        const response = d.agent_response || "";
        const truncated = response.length > RESPONSE_PREVIEW_LENGTH && !isExpanded;
        const displayText = truncated
          ? response.slice(0, RESPONSE_PREVIEW_LENGTH) + "…"
          : response;

        return (
          <div key={d.id} style={styles.card}>

            {/* header */}
            <div style={styles.cardHeader}>
              <div style={styles.headerLeft}>
                {d.anomaly_types.map((type, i) => (
                  <span key={i} style={styles.anomalyTag}>
                    {typeof type === "string" ? type : type.type}
                  </span>
                ))}
              </div>
              <div style={styles.headerRight}>
                <span style={{ ...styles.fixedBadge, backgroundColor: badge.color }}>
                  {badge.label}
                </span>
                <span style={styles.timestamp}>{formatTime(d.timestamp)}</span>
              </div>
            </div>

            {/* tools used */}
            <div style={styles.toolsRow}>
              {d.tools_used.map((tool) => (
                <span key={tool} style={styles.toolTag}>
                  {tool}
                </span>
              ))}
            </div>

            {/* agent response with expand/collapse */}
            <pre style={styles.response}>{displayText}</pre>
            {response.length > RESPONSE_PREVIEW_LENGTH && (
              <button style={styles.expandBtn} onClick={() => toggleExpanded(d.id)}>
                {isExpanded ? "show less" : "show more"}
              </button>
            )}

          </div>
        );
      })}
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
    marginBottom: "16px",
    marginTop: 0,
  },
  card: {
    backgroundColor: "#1a1a1a",
    borderRadius: "8px",
    padding: "16px",
    marginBottom: "12px",
    border: "1px solid #2a2a2a",
  },
  cardHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "10px",
  },
  headerLeft: {
    display: "flex",
    gap: "6px",
    flexWrap: "wrap",
  },
  headerRight: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
  },
  anomalyTag: {
    padding: "3px 8px",
    borderRadius: "4px",
    backgroundColor: "#2a2a2a",
    color: "#f59e0b",
    fontSize: "11px",
    fontWeight: 500,
  },
  fixedBadge: {
    padding: "3px 10px",
    borderRadius: "999px",
    fontSize: "11px",
    color: "#fff",
    fontWeight: 500,
  },
  timestamp: {
    fontSize: "11px",
    color: "#555",
  },
  toolsRow: {
    display: "flex",
    gap: "6px",
    flexWrap: "wrap",
    marginBottom: "10px",
  },
  toolTag: {
    padding: "2px 8px",
    borderRadius: "4px",
    backgroundColor: "#0f1f3d",
    color: "#3b82f6",
    fontSize: "11px",
    fontFamily: "monospace",
  },
  response: {
    fontSize: "12px",
    color: "#aaa",
    whiteSpace: "pre-wrap",
    wordBreak: "break-word",
    margin: 0,
    lineHeight: 1.6,
    borderTop: "1px solid #2a2a2a",
    paddingTop: "10px",
  },
  expandBtn: {
    marginTop: "8px",
    background: "none",
    border: "none",
    color: "#3b82f6",
    fontSize: "11px",
    cursor: "pointer",
    padding: 0,
  },
  empty: {
    color: "#555",
    fontSize: "14px",
    padding: "40px 0",
    textAlign: "center",
  },
};
