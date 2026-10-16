export default function RecoveryReplay({ replay }) {
  if (!replay) return null;

  return (
    <section className="recovery-replay" aria-labelledby="recovery-title">
      <div className="recovery-heading">
        <div>
          <h2 id="recovery-title">Checkpointed recovery replay</h2>
          <p>LangGraph workflow, fixture model and tool · source SHA-256 {replay.source_sha256.slice(0, 12)}</p>
        </div>
        <span>{replay.resumed_status}</span>
      </div>
      <ol>
        <li><strong>First run</strong><code>{replay.actions[0]}(last_n=2)</code></li>
        <li><strong>Checkpoint</strong><code>{replay.first_status} at budget boundary</code></li>
        <li><strong>Resume</strong><code>same thread: {replay.thread_id}</code></li>
        <li><strong>Resumed run</strong><code>{replay.actions[1]} → {replay.resumed_status}</code></li>
      </ol>
      <p className="recovery-scope">{replay.scope}</p>
    </section>
  );
}
