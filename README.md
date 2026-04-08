---
title: MIPS Pipeline Instruction Scheduler
emoji: ⚡
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# MIPS Pipeline Instruction Scheduling Environment

An OpenEnv-compliant reinforcement learning environment where agents learn to **schedule MIPS assembly instructions** to minimize pipeline stalls in a simulated 5-stage pipelined processor.

## Motivation

Modern compilers face the NP-hard problem of ordering thousands of assembly instructions to minimize execution stalls on pipelined processors. The dependencies between instructions form a **Directed Acyclic Graph (DAG)**, and finding the optimal execution order requires understanding hardware-specific quirks like forwarding paths, load-use hazards, and branch prediction.

This environment models this real-world compiler optimization task, allowing RL agents to learn scheduling strategies that outperform hand-crafted heuristics.

## Pipeline Model

The simulator models a **classic 5-stage MIPS pipeline**:

```
IF → ID → EX → MEM → WB
```

### Features
| Feature | Implementation |
|---------|---------------|
| **Forwarding** | EX→EX and MEM→EX bypass paths |
| **Register File** | Half-cycle: WB writes 1st half, ID reads 2nd half (same-cycle OK) |
| **Load-Use Hazard** | 1-cycle stall when lw is immediately followed by dependent instruction |
| **Structural Hazards** | Single memory port; tracked for completeness |
| **Branch Prediction** | 2-bit saturating counter, branch resolved at EX → 2-cycle flush on mispredict |
| **Dependencies** | RAW, WAR, WAW register deps + memory ordering (store→load, store→store, load→store) |

## Action Space

```python
class SchedulerAction(Action):
    instruction_id: int  # Must be in current legal_actions
```

The agent picks one instruction at a time from a **dynamically masked list** of currently schedulable instructions (all predecessors in the dependency DAG have been scheduled).

## Observation Space

```python
class SchedulerObservation(Observation):
    # DAG structure (constant per episode)
    num_instructions: int
    instructions: List[dict]     # [{id, opcode, dest_reg, src_regs, type, raw_text}]
    edges: List[List[Any]]       # [[from_id, to_id, dep_type, register], ...]

    # Current state
    scheduled_ids: List[int]     # Already scheduled instruction IDs (in order)
    legal_actions: List[int]     # Currently schedulable IDs

    # Pipeline feedback
    stalls_this_step: int        # Stalls caused by last scheduled instruction
    total_stalls: int            # Cumulative stalls
    total_cycles: int            # Total pipeline cycles

    # Inherited: done, reward, metadata
```

## Tasks

| Task | Difficulty | Instructions | Types | Description |
|------|-----------|-------------|-------|-------------|
| `easy_alu_chain` | Easy | 10 | R-type + addi | Simple ALU chains, no memory/branches |
| `medium_memory_mix` | Medium | 20 | R + lw/sw + beq | Load-use hazards + branch prediction |
| `hard_full_block` | Hard | 58 | Full MIPS subset | Matrix ops, many loads, complex deps + branch |

## Reward Function

- **Per-step**: `−stalls_this_step` (penalizes stall-inducing orderings)
- **Episode-end bonus**: `exp(−total_stalls / max_possible_stalls)` (rewards low total stalls)

## Grading

Scores are in `[0.0, 1.0]`:
- **1.0** = matches or beats the greedy-best schedule (optimal)
- **0.0** = equal or worse than naive program order (worst)
- Linear interpolation between these bounds

## Quick Start

### Using Docker

```bash
# Build the image from the environment root
docker build -t mips-scheduler-env:latest .

# Run the Easy task
docker run -p 8000:8000 -e MIPS_SCHEDULER_TASK=easy_alu_chain mips-scheduler-env:latest

# Run the Medium task
docker run -p 8000:8000 -e MIPS_SCHEDULER_TASK=medium_memory_mix mips-scheduler-env:latest

# Run the Hard task
docker run -p 8000:8000 -e MIPS_SCHEDULER_TASK=hard_full_block mips-scheduler-env:latest
```

### Using Python Client

```python
import asyncio
from mips_scheduler_env import MIPSSchedulerEnv, SchedulerAction

async def main():
    env = await MIPSSchedulerEnv.from_docker_image("mips-scheduler-env:latest")
    async with env:
        result = await env.reset(task_name="easy_alu_chain")
        while not result.done:
            # Pick first legal action (naive baseline)
            inst_id = result.observation.legal_actions[0]
            result = await env.step(SchedulerAction(instruction_id=inst_id))
        print(f"Score: {result.observation.metadata['final_grade']}")

asyncio.run(main())
```

### Synchronous Usage

```python
from mips_scheduler_env import MIPSSchedulerEnv, SchedulerAction

# Connect to the running container
with MIPSSchedulerEnv(base_url="http://localhost:8000").sync() as env:
    # You can select ANY task here: 
    # "easy_alu_chain", "medium_memory_mix", or "hard_full_block"
    result = env.reset(task_name="hard_full_block")
    
    while not result.done:
        inst_id = result.observation.legal_actions[0]
        result = env.step(SchedulerAction(instruction_id=inst_id))
        
    print(f"Grade: {result.observation.metadata['final_grade']}")
```

## Evaluating All Tasks

To run the full hackathon evaluation script which benchmarks your agent across all three tasks (Easy, Medium, and Hard), run the `inference.py` script provided in the root:

```bash
# Set your API credentials
export HF_TOKEN="your_huggingface_token"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"

# Run the baseline evaluation
python inference.py
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MIPS_SCHEDULER_TASK` | Task to load on reset | `easy_alu_chain` |
| `API_BASE_URL` | LLM API endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Model identifier | `Qwen/Qwen2.5-72B-Instruct` |
| `HF_TOKEN` | API key | (required) |

## Baseline Scores

| Task | Naive (program order) | Greedy Best | LLM Baseline |
|------|-----------------------|-------------|--------------|
| `easy_alu_chain` | ~3 stalls | 0 stalls | TBD |
| `medium_memory_mix` | ~8 stalls | ~2 stalls | TBD |
| `hard_full_block` | ~15 stalls | ~4 stalls | TBD |

## Project Structure

```
mips_scheduler_env/
├── __init__.py                  # Package exports
├── models.py                    # SchedulerAction, SchedulerObservation, SchedulerState
├── client.py                    # MIPSSchedulerEnv (EnvClient subclass)
├── openenv.yaml                 # OpenEnv spec metadata
├── pyproject.toml               # Python package config
├── README.md                    # This file
└── server/
    ├── __init__.py              # Server exports
    ├── app.py                   # FastAPI application
    ├── mips_scheduler_env.py    # Environment (reset/step/state)
    ├── pipeline_simulator.py    # 5-stage MIPS pipeline simulation
    ├── dag_generator.py         # Assembly parser + dependency DAG builder
    ├── tasks.py                 # Task definitions (easy/medium/hard)
    ├── graders.py               # Deterministic grading functions
    └── Dockerfile               # Container image
```

## How It Works

1. **Assembly → Parse**: Raw MIPS assembly text is parsed into structured `MIPSInstruction` objects
2. **Dependency Analysis**: RAW, WAR, WAW register dependencies + memory ordering are identified
3. **DAG Construction**: Instructions become nodes, dependencies become edges
4. **Kahn's Algorithm**: At each step, instructions with all predecessors scheduled are "legal"
5. **Agent Chooses**: The agent picks which legal instruction to place next
6. **Pipeline Simulation**: The chosen order is simulated cycle-by-cycle to count stalls
7. **Grading**: Final stall count is compared against baselines → score [0, 1]
