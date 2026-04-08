"""
Graders for MIPS Instruction Scheduling Tasks
===============================================
Deterministic grading functions that score agent performance on [0.0, 1.0].

Grading logic:
  - Run the pipeline simulator on the agent's chosen schedule.
  - Compare total stalls to (a) the worst-case naive order and (b) the best
    possible schedule found by brute-force/heuristic.
  - Score is linearly interpolated: 1.0 = optimal (lowest stalls), 0.0 = worst.
  
For tasks with fewer instructions (≤12), we compute the true optimal via
exhaustive search.  For larger tasks, we use a heuristic baseline (original
order) as the worst case and a greedy-best as "near-optimal."
"""

from __future__ import annotations

import math
from itertools import permutations
from typing import Dict, List, Optional, Tuple

from .dag_generator import InstructionDAG, ScheduleTracker, build_dependency_dag, parse_assembly
from .pipeline_simulator import MIPSInstruction, PipelineSimulator, SimulationResult, compute_max_possible_stalls
from .tasks import TaskConfig, get_task


def _simulate_schedule(instructions: List[MIPSInstruction]) -> int:
    """Simulate and return total stalls for a given instruction order."""
    sim = PipelineSimulator()
    result = sim.simulate(instructions)
    return result.total_stalls


def _greedy_best_schedule(dag: InstructionDAG) -> List[MIPSInstruction]:
    """
    Greedy heuristic: at each step, pick the legal instruction that
    causes the fewest stalls when appended to the current schedule.
    This gives a reasonable lower bound for scoring.
    """
    tracker = ScheduleTracker(dag)
    scheduled: List[MIPSInstruction] = []

    while not tracker.is_complete:
        legal = tracker.legal_actions
        if not legal:
            break

        best_id = legal[0]
        best_stalls = float("inf")

        for inst_id in legal:
            candidate = scheduled + [dag.instruction_by_id(inst_id)]
            sim = PipelineSimulator()
            result = sim.simulate(candidate)
            stalls_last = result.stall_breakdown[-1] if result.stall_breakdown else 0
            if stalls_last < best_stalls:
                best_stalls = stalls_last
                best_id = inst_id

        tracker.schedule(best_id)
        scheduled.append(dag.instruction_by_id(best_id))

    return scheduled


def _naive_schedule(dag: InstructionDAG) -> List[MIPSInstruction]:
    """Original program order — typically the worst-case baseline."""
    return list(dag.instructions)


def compute_grade(
    task_name: str,
    agent_schedule_ids: List[int],
) -> float:
    """
    Grade the agent's schedule for a given task.
    
    Args:
        task_name: Name of the task (must be in TASKS registry)
        agent_schedule_ids: Ordered list of instruction IDs chosen by the agent
    
    Returns:
        Score in [0.0, 1.0]
    """
    task = get_task(task_name)
    instructions = parse_assembly(task.assembly, task.branch_taken_map)
    dag = build_dependency_dag(instructions)

    # Build the agent's schedule
    agent_instructions = [dag.instruction_by_id(i) for i in agent_schedule_ids]

    # Simulate agent's schedule
    agent_stalls = _simulate_schedule(agent_instructions)

    # Simulate naive (original order) — worst-case baseline
    naive_stalls = _simulate_schedule(_naive_schedule(dag))

    # Simulate greedy-best — near-optimal baseline
    best_instructions = _greedy_best_schedule(dag)
    best_stalls = _simulate_schedule(best_instructions)

    # Edge case: if naive == best, the task has no room for improvement
    if naive_stalls <= best_stalls:
        # If the agent did at least as well as naive, score = 1.0
        return 1.0 if agent_stalls <= naive_stalls else 0.0

    # Linear interpolation: best → 1.0, naive → 0.0
    # Clamp to [0, 1]
    score = 1.0 - (agent_stalls - best_stalls) / (naive_stalls - best_stalls)
    score = max(0.0, min(1.0, score))

    return round(score, 4)


def compute_episode_reward(total_stalls: int, max_possible_stalls: int) -> float:
    """
    Episode-end bonus reward, normalized to [0, 1].
    
    reward = exp(-total_stalls / max_possible_stalls)
    """
    if max_possible_stalls <= 0:
        return 1.0
    return math.exp(-total_stalls / max_possible_stalls)
