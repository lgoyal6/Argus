import logger
import obs

# set this to your actual run ID from Supabase
RUN_ID = "26a75d76-72f9-4afe-abc3-fd1474284d0f"  # e.g. "your-uuid-here"

import os
import time
import json
import requests
from pathlib import Path
from detector import detect_anomalies, newest_trace_context
from tools import TOOLS, run_tool
from prompts import SYSTEM_PROMPT, build_user_prompt
from logger import log_decision
from attempt import Attempt, TIMED_OUT, reconcile_open_attempts, resume_attempt
import anthropic

try:  # imported as a package by the tests, flat by the agent image
    from backend.argus_secrets import get_secret
except ImportError:  # pragma: no cover - exercised only by the flat layout
    from argus_secrets import get_secret

from dotenv import load_dotenv
load_dotenv("../.env")
# ── config ─────────────────────────────────────────────────────────────────────
BACKEND_URL = os.environ.get("ARGUS_BACKEND_URL", "http://backend:8000")
METRICS_FILE = "../training_job/metrics/metrics.jsonl"
CONFIG_PATH = "../training_job/config.yaml"
TRAINING_DIR = "../training_job"
POLL_INTERVAL = 10        # seconds between each check
MAX_TOOL_ROUNDS = 10      # max tool calls per agent invocation
COOLDOWN_STEPS = 500      # steps to wait before re-triggering the same anomaly type


# ── agent call ─────────────────────────────────────────────────────────────────
def run_agent(anomalies):
    # One attempt record per invocation, opened before anything is asked for. The
    # trigger is recorded now so the claim at the end can be checked against what
    # actually set it off, and the metrics watermark is captured before any action so
    # "new rows" means new relative to the request rather than to the reply.
    attempt = resume_attempt(RUN_ID, anomalies[0])
    resumed = attempt is not None
    if attempt is None:
        attempt = Attempt(run_id=RUN_ID, trigger=anomalies[0])
        attempt.capture_baseline(METRICS_FILE)
    elif attempt.baseline is None:
        attempt.capture_baseline(METRICS_FILE)
    else:
        # Re-derive the rung from durable metrics before asking for another
        # action. A trainer may have progressed while the agent was down.
        attempt.observe(METRICS_FILE)

    if attempt.fixed:
        final_text = (
            f"Recovered attempt {attempt.attempt_id}; durable metrics already "
            "show that the triggering anomaly was corrected."
        )
        log_decision(
            anomalies=anomalies,
            agent_response=final_text,
            tools_used=[],
            status=attempt.status(),
            attempt=attempt.to_dict(),
        )
        return final_text

    # Resolve the model key only when more action is still needed. A recovered
    # terminal attempt must not pay for or dispatch another model/tool round.
    client = anthropic.Anthropic(api_key=get_secret("ANTHROPIC_API_KEY"))

    user_prompt = build_user_prompt(
        anomalies=anomalies,
        metrics_file=METRICS_FILE,
        config_path=CONFIG_PATH,
        training_dir=TRAINING_DIR
    )
    if resumed:
        user_prompt += (
            "\n\nResume recovery attempt " + attempt.attempt_id + ". Its last "
            "durable action was " + json.dumps(attempt.approved_action) +
            ". Check current evidence before repeating an action."
        )

    messages = [{"role": "user", "content": user_prompt}]

    print("\n── agent invoked ──────────────────────────────────────────")
    print(f"anomalies: {[a['type'] for a in anomalies]}")

    all_tools_used = []

    for round_num in range(MAX_TOOL_ROUNDS):
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages
        )

        # collect text from response
        text_blocks = [b.text for b in response.content if hasattr(b, "text")]
        tool_blocks = [b for b in response.content if b.type == "tool_use"]

        if text_blocks:
            print(f"\nagent: {''.join(text_blocks)}")

        # if no tool calls, agent is done
        if not tool_blocks or response.stop_reason == "end_turn":
            print("\n── agent finished ─────────────────────────────────────────")
            final_text = "\n".join(text_blocks)
            # The status comes from what was observed in the metrics stream, not from
            # which tools returned without an error. observe() is what decides whether
            # a trainer ran at all.
            attempt.observe(METRICS_FILE)
            print(f"attempt {attempt.attempt_id}: {attempt.outcome} ({attempt.terminal})")
            log_decision(
                anomalies=anomalies,
                agent_response=final_text,
                # Only tools that were actually dispatched. Names lifted off an
                # undispatched block would record work that never happened.
                tools_used=all_tools_used,
                status=attempt.status(),
                attempt=attempt.to_dict()
            )
            return final_text

        # process tool calls
        tool_results = []
        for tool_call in tool_blocks:
            # The tool input is model-chosen and the result of read_config is the whole
            # config file. Printing either put configuration contents into the container
            # log; obs.log keeps the tool name, which is the part anyone reads, and
            # summarises the rest.
            print(f"\ncalling tool: {tool_call.name}")
            obs.log("recovery.tool_call", tool=tool_call.name, payload=tool_call.input)

            result = run_tool(tool_call.name, tool_call.input,
                              attempt_id=attempt.attempt_id)
            obs.log("recovery.tool_result", tool=tool_call.name,
                    refused=bool(isinstance(result, dict) and result.get("refused")),
                    payload=result)

            all_tools_used.append(tool_call.name)

            # The approved action is the patch the sandbox actually applied, which is
            # not always the patch the model asked for.
            if tool_call.name == "patch_config" and result.get("status") == "patched":
                attempt.record_action(
                    {"tool": "patch_config", "changes": result.get("changes")},
                    resulting_config=run_tool("read_config", {"config_path": CONFIG_PATH})
                )

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": json.dumps(result)
            })

        # append assistant response and tool results to message history
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    print("max tool rounds reached")
    # Running out of rounds is a timeout, not a failure to act: the difference matters
    # to whether retrying the same thing is sensible.
    attempt.observe(METRICS_FILE)
    attempt.fail(TIMED_OUT, reason="max tool rounds reached")
    log_decision(
        anomalies=anomalies,
        agent_response="Max tool rounds reached without resolution.",
        tools_used=all_tools_used,
        status=attempt.status(),
        attempt=attempt.to_dict()
    )
    return None


def poll_once(last_handled, act=None):
    """One cycle: ingest what is new, look at it, and act if it is wrong.

    One span per poll, parented on the newest row the trainer wrote. That row is what
    this poll exists to act on and it arrived from another process, so it is the only
    available parent - and taking it here means the ingest call, the detection and
    any recovery below are one trace rather than three unrelated ones that no query
    joins.

    `act` defaults to the real agent invocation. It is a parameter because that
    invocation is a paid model call, which makes the whole cycle undrivable without
    one; nothing else about the cycle changes.
    """
    act = act or run_agent
    with obs.span("agent.poll", kind="consumer",
                  parent=newest_trace_context(METRICS_FILE)):
        # Inside the poll span so the backend's server span, and the database write
        # under it, join this trace rather than starting one of their own.
        with obs.span("agent.ingest_request", kind="client", dependency="backend"):
            headers = {}
            traceparent = obs.current_traceparent()
            if traceparent:
                headers["traceparent"] = traceparent
            try:
                requests.post(
                    f"{BACKEND_URL}/runs/{RUN_ID}/metrics/sync",
                    json={"metrics_file": METRICS_FILE},
                    headers=headers,
                    timeout=5
                )
            except Exception:
                pass

        # A worker may die after dispatching a recovery while the trainer keeps
        # writing. Reconcile durable open attempts on the next worker's polls so
        # the original attempt reaches its evidence-backed terminal state.
        reconcile_open_attempts(RUN_ID, METRICS_FILE)

        anomalies = detect_anomalies(METRICS_FILE)
        if not anomalies:
            print(".", end="", flush=True)
            return []

        # only act if this anomaly type hasn't been handled within the last COOLDOWN_STEPS
        new_anomalies = [
            a for a in anomalies
            if a["step"] - last_handled.get(a["type"], -COOLDOWN_STEPS) >= COOLDOWN_STEPS
        ]
        if not new_anomalies:
            print(".", end="", flush=True)
            return []
        for a in new_anomalies:
            last_handled[a["type"]] = a["step"]
        act(new_anomalies)
        return new_anomalies


# ── main loop ──────────────────────────────────────────────────────────────────
def main():
    print("Argus agent started")
    # No-op unless OTEL_EXPORTER_OTLP_ENDPOINT is set.
    obs.setup_tracing("argus-agent")
    if RUN_ID:
        logger.set_run_id(RUN_ID)
    print(f"watching: {METRICS_FILE}")
    print(f"polling every {POLL_INTERVAL}s\n")

    last_handled = {}  # maps anomaly type -> step it was last handled at

    while True:
        poll_once(last_handled)
        # Outside the poll span: a span that covered the sleep would report the poll
        # interval as work.
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
