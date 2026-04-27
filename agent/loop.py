import logger

# set this to your actual run ID from Supabase
RUN_ID = "26a75d76-72f9-4afe-abc3-fd1474284d0f"  # e.g. "your-uuid-here"

import time
import json
from pathlib import Path
from detector import detect_anomalies
from tools import TOOLS, run_tool
from prompts import SYSTEM_PROMPT, build_user_prompt
from logger import log_decision
import anthropic
from dotenv import load_dotenv
load_dotenv("../.env")
# ── config ─────────────────────────────────────────────────────────────────────
METRICS_FILE = "../training_job/metrics/metrics.jsonl"
CONFIG_PATH = "../training_job/config.yaml"
TRAINING_DIR = "../training_job"
POLL_INTERVAL = 10        # seconds between each check
MAX_TOOL_ROUNDS = 10      # max tool calls per agent invocation


# ── agent call ─────────────────────────────────────────────────────────────────
def run_agent(anomalies):
    client = anthropic.Anthropic()

    user_prompt = build_user_prompt(
        anomalies=anomalies,
        metrics_file=METRICS_FILE,
        config_path=CONFIG_PATH,
        training_dir=TRAINING_DIR
    )

    messages = [{"role": "user", "content": user_prompt}]

    print("\n── agent invoked ──────────────────────────────────────────")
    print(f"anomalies: {[a['type'] for a in anomalies]}")

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
            log_decision(
                anomalies=anomalies,
                agent_response=final_text,
                tools_used=[b.name for b in tool_blocks] if tool_blocks else []
            )
            return final_text

        # process tool calls
        tool_results = []
        for tool_call in tool_blocks:
            print(f"\ncalling tool: {tool_call.name}")
            print(f"input: {json.dumps(tool_call.input, indent=2)}")

            result = run_tool(tool_call.name, tool_call.input)
            print(f"result: {json.dumps(result, indent=2)[:500]}")

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": json.dumps(result)
            })

        # append assistant response and tool results to message history
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    print("max tool rounds reached")
    return None


# ── main loop ──────────────────────────────────────────────────────────────────
def main():
    print("AutoDebug agent started")
    if RUN_ID:
        logger.set_run_id(RUN_ID)
    print(f"watching: {METRICS_FILE}")
    print(f"polling every {POLL_INTERVAL}s\n")

    seen_steps = set()

    while True:
        anomalies = detect_anomalies(METRICS_FILE)

        if anomalies:
            # only act on anomalies we haven't seen before
            new_anomalies = [
                a for a in anomalies
                if a["step"] not in seen_steps
            ]

            if new_anomalies:
                for a in new_anomalies:
                    seen_steps.add(a["step"])
                run_agent(new_anomalies)
            else:
                print(".", end="", flush=True)
        else:
            print(".", end="", flush=True)

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
