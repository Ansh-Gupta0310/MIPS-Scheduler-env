"""
Inference Script — MIPS Pipeline Instruction Scheduling
========================================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- The inference script must be named `inference.py` and placed in the root directory.
- Participants must use OpenAI Client for all LLM calls using above variables.

STDOUT FORMAT
- The script emits exactly three line types to stdout:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import os
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ── OpenEnv imports (from the environment package) ──────────────────────────
# These are installed via pip install ./envs/mips_scheduler_env
from mips_scheduler_env import MIPSSchedulerEnv, SchedulerAction

# ── Configuration ───────────────────────────────────────────────────────────

IMAGE_NAME = os.getenv("IMAGE_NAME", "mips-scheduler-env:latest")  # Docker image for the env
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = "mips_scheduler"

MAX_STEPS = 60
TEMPERATURE = 0.3
MAX_TOKENS = 100

TASKS = ["easy_alu_chain", "medium_memory_mix", "hard_full_block"]

SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert MIPS compiler optimizing instruction schedules for a
5-stage pipelined processor (IF→ID→EX→MEM→WB) with data forwarding and
a 2-bit branch predictor.

Your goal: minimize pipeline stalls by choosing the best instruction to
schedule next from the legal_actions list.

Key hazards to avoid:
- Load-use: scheduling an instruction that reads a register written by a
  load (lw) immediately before it causes a 1-cycle stall.
- RAW data hazard: scheduling an instruction that reads a register written
  by the immediately previous instruction may cause stalls if forwarding
  cannot resolve it.
- Branch misprediction: causes a 2-cycle flush.

Strategy: separate loads from their consumers, interleave independent
instructions to fill hazard slots.

Reply with ONLY a single integer — the instruction_id from legal_actions.
No explanation, no text, just the number.
""").strip()


# ── Logging functions ───────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Build prompt from observation ───────────────────────────────────────────

def build_user_prompt(obs: Any, step: int, history: List[str]) -> str:
    """Build a prompt showing the DAG, current schedule, and legal actions."""
    # Format instructions
    inst_lines = []
    for inst in obs.instructions:
        deps = [
            f"{e[0]}→{e[1]}({e[2]}:{e[3]})"
            for e in obs.edges
            if e[1] == inst["id"]
        ]
        dep_str = ", ".join(deps) if deps else "none"
        scheduled_mark = "✓" if inst["id"] in obs.scheduled_ids else " "
        inst_lines.append(
            f"  [{scheduled_mark}] id={inst['id']:2d}  {inst['raw_text']:30s}  deps: {dep_str}"
        )

    history_block = "\n".join(history[-5:]) if history else "None"

    return textwrap.dedent(f"""\
Step: {step}
Task: {obs.task_name} ({obs.difficulty})
Total stalls so far: {obs.total_stalls}
Stalls last step: {obs.stalls_this_step}

Instructions:
{chr(10).join(inst_lines)}

Already scheduled (in order): {obs.scheduled_ids}
Legal actions (pick one): {obs.legal_actions}

Recent history:
{history_block}

Reply with ONLY a single integer from the legal_actions list.
""").strip()


# ── Parse model response ───────────────────────────────────────────────────

def parse_model_response(response_text: str, legal_actions: List[int]) -> int:
    """
    Extract an integer instruction_id from the model's response.
    Specifically looks for the final number mentioned if the model 
    provided reasoning, or any number that matches a legal action.
    """
    import re
    
    # Remove markdown bold/code formatting
    clean_text = response_text.replace("**", "").replace("`", "")
    
    # Find all integers in the text
    numbers = re.findall(r"\d+", clean_text)
    
    if not numbers:
        return legal_actions[0] if legal_actions else 0

    # Strategy: Check numbers from last to first (final answers are usually at the end)
    for num_str in reversed(numbers):
        num = int(num_str)
        if num in legal_actions:
            return num
            
    # Fallback to first mentioned legal number
    for num_str in numbers:
        num = int(num_str)
        if num in legal_actions:
            return num

    return legal_actions[0] if legal_actions else 0


# ── Main loop ───────────────────────────────────────────────────────────────

async def run_task(task_name: str, client: OpenAI) -> float:
    """Run one task and return the score."""
    env = await MIPSSchedulerEnv.from_docker_image(IMAGE_NAME)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_name=task_name)
        obs = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            if not obs.legal_actions:
                break

            # Build prompt
            user_prompt = build_user_prompt(obs, step, history)

            # Query LLM
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=False,
                )
                response_text = (completion.choices[0].message.content or "").strip()
            except Exception as exc:
                print(f"[DEBUG] Model request failed: {exc}", flush=True)
                response_text = str(obs.legal_actions[0])

            # Parse action
            inst_id = parse_model_response(response_text, obs.legal_actions)

            # Step
            result = await env.step(SchedulerAction(instruction_id=inst_id))
            obs = result.observation

            reward = result.reward or 0.5
            # Clamp reward to (0.01, 0.99) — validator requires all values in this range
            reward = max(0.01, min(0.99, float(reward)))
            done = result.done
            error = None  # metadata is stripped by serialization, don't try to read it

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=str(inst_id),
                reward=reward,
                done=done,
                error=error,
            )

            history.append(f"Step {step}: scheduled inst {inst_id} → stalls {obs.stalls_this_step}")

            if done:
                break

        # Extract final grade — must be strictly in (0, 1) per validator rules
        score = getattr(obs, 'final_grade', 0.01)
        # Safety clamp: NEVER exactly 0.0 or 1.0
        score = max(0.01, min(0.99, float(score)))
        success = True

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


async def main() -> None:
    """Run all 3 tasks and report scores."""
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print("=" * 60, flush=True)
    print("MIPS Instruction Scheduling — Baseline Inference", flush=True)
    print("=" * 60, flush=True)

    scores = {}
    for task_name in TASKS:
        print(f"\n{'─' * 40}", flush=True)
        print(f"Running task: {task_name}", flush=True)
        print(f"{'─' * 40}", flush=True)
        score = await run_task(task_name, client)
        scores[task_name] = score

    print(f"\n{'=' * 60}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'=' * 60}", flush=True)
    for task, s in scores.items():
        print(f"  {task:25s}  score={s:.3f}", flush=True)
    avg = sum(scores.values()) / len(scores)
    print(f"  {'AVERAGE':25s}  score={avg:.3f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
