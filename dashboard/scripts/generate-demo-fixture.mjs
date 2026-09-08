import { spawnSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const dashboardDir = fileURLToPath(new URL("..", import.meta.url));
const rootDir = join(dashboardDir, "..");
const walkthroughPath = join(rootDir, "examples", "offline_walkthrough.py");
const metricsPath = join(rootDir, "examples", "sample_metrics.jsonl");
const outputPath = process.argv[2] || join(dashboardDir, "public", "demo-data.json");

const result = spawnSync("python3", [walkthroughPath, metricsPath], {
  cwd: rootDir,
  encoding: "utf8",
});

if (result.status !== 0) {
  process.stderr.write(result.stderr || result.stdout);
  process.exit(result.status ?? 1);
}

const detectorReport = JSON.parse(result.stdout);
const metrics = readFileSync(metricsPath, "utf8")
  .split("\n")
  .filter(Boolean)
  .map((line) => JSON.parse(line));
const lastStep = metrics.at(-1)?.step;
const runId = "bundled-detector-replay";

const payload = {
  metadata: {
    label: "Bundled demo data",
    source: "examples/sample_metrics.jsonl",
    rows: detectorReport.rows,
    scope:
      "Read-only detector replay. No model API, backend, Supabase, or repair action runs in this public demo.",
  },
  runs: [
    {
      id: runId,
      name: "bundled anomaly replay",
      status: "completed",
    },
  ],
  metrics: {
    [runId]: metrics,
  },
  decisions: {
    [runId]: [
      {
        id: "bundled-detector-findings",
        run_id: runId,
        timestamp: null,
        anomaly_types: detectorReport.detected.map((type) => ({
          type,
          step: lastStep,
        })),
        tools_used: ["read_metrics", "detect_anomalies"],
        agent_response:
          "Argus replayed the bundled training trace through its production detector and found the expected loss spike and gradient explosion at step 21. This public replay stops at detection; the credentialed agent repair loop is documented in the repository.",
        fixed: null,
        status: "detected",
      },
    ],
  },
};

mkdirSync(dirname(outputPath), { recursive: true });
writeFileSync(outputPath, `${JSON.stringify(payload, null, 2)}\n`);
console.log(`generated ${outputPath} from ${detectorReport.rows} detector rows`);
