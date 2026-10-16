import { useRef, useState } from "react";

const example = "0.95, 1, 1.05, 1, 0.95, 1, 1.05, 1, 0.95, 1, 1.05, 1, 0.95, 1, 1.05, 1, 0.95, 1, 1.05, 1, 1.25";

export default function DetectorPlayground() {
  const revision = useRef(0);
  const pending = useRef(null);
  const [history, setHistory] = useState(example);
  const [result, setResult] = useState(null);

  async function run(event) {
    event.preventDefault();
    const current = ++revision.current;
    pending.current?.abort();
    const controller = new AbortController();
    pending.current = controller;
    const timer = setTimeout(() => controller.abort(), 12000);
    setResult({ loading: true, error: null, finding: null, count: 0 });
    try {
      const response = await fetch("/api/argus_detect", {
        method: "POST",
        signal: controller.signal,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ history }),
      });
      const result = await response.json();
      if (current !== revision.current) return;
      if (!response.ok) throw new Error(result.error || "The detector could not evaluate this input.");
      setResult({ error: null, finding: result.finding, count: result.count });
    } catch (error) {
      if (current === revision.current) setResult({ error: error.message, finding: null, count: 0 });
    } finally {
      clearTimeout(timer);
    }
  }

  return (
    <section className="detector-playground" aria-labelledby="detector-title">
      <div className="detector-copy">
        <p className="detector-eyebrow">RUN THE DETERMINISTIC DETECTOR</p>
        <h2 id="detector-title">Try your own gradient history</h2>
        <p>
          Enter up to 21 gradient norms. Argus checks the latest value against the
          existing hard threshold and rolling z-score rule.
        </p>
      </div>
      <form onSubmit={run}>
        <label htmlFor="gradient-history">Gradient norms, oldest to newest</label>
        <textarea
          id="gradient-history"
          value={history}
          onChange={(event) => { revision.current++; pending.current?.abort(); setHistory(event.target.value); setResult(null); }}
          rows={3}
          maxLength={512}
          aria-describedby="gradient-help"
        />
        <div className="detector-form-row">
          <span id="gradient-help">Comma or line separated, max 21 values, each from 0 to 1,000,000.</span>
          <button type="submit" disabled={result?.loading}>Run detector</button>
        </div>
      </form>
      {result && (
        <output className={`detector-result ${result.error ? "is-error" : ""}`} aria-live="polite">
          {result.loading
            ? "Running the source detector..."
            : result.error
            ? result.error
            : result.finding
              ? `Flagged at step ${result.finding.step}: ${result.finding.description}`
              : `No finding across ${result.count} samples.`}
        </output>
      )}
      <p className="detector-limit">
        This sends only the bounded metric history to the deterministic Python detector.
        It does not start training,
        call a model, make a repair, or rerun a job. A finding is a signal, not a diagnosis.
      </p>
    </section>
  );
}
