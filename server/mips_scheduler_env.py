"""
MIPS Instruction Scheduling Environment
=========================================
OpenEnv-compliant environment that lets an RL agent schedule MIPS assembly
instructions to minimize pipeline stalls.

reset()  → Generates a DAG from a task's assembly code.
step(action) → Agent picks the next instruction to place.
state    → Current episode metadata.
"""

from __future__ import annotations

import math
import os
import uuid
from typing import Any, Optional

from openenv.core.env_server.interfaces import Action, Environment, Observation

from ..models import SchedulerAction, SchedulerObservation, SchedulerState
from .dag_generator import build_dependency_dag, parse_assembly, ScheduleTracker
from .graders import compute_episode_reward, compute_grade
from .pipeline_simulator import (
    MIPSInstruction,
    PipelineSimulator,
    compute_max_possible_stalls,
)
from .tasks import get_task, list_tasks, TaskConfig


class MIPSSchedulerEnvironment(
    Environment[SchedulerAction, SchedulerObservation, SchedulerState]
):
    """
    An OpenEnv environment for MIPS pipeline instruction scheduling.

    The agent receives a DAG of MIPS instructions and must choose a valid
    topological ordering that minimizes stalls in a 5-stage pipeline.
    """

    def __init__(self):
        super().__init__()
        self._state = SchedulerState()
        self._task: Optional[TaskConfig] = None
        self._tracker: Optional[ScheduleTracker] = None
        self._simulator: Optional[PipelineSimulator] = None
        self._scheduled_instructions: list[MIPSInstruction] = []
        self._dag_dict: dict = {}
        self._max_possible_stalls: int = 1
        self._total_stalls: int = 0
        self._total_cycles: int = 0
        self._schedule_order: list[int] = []  # IDs in scheduling order

    # ── reset ──────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SchedulerObservation:
        """
        Reset the environment for a new episode.

        Reads the task name from:
          1. kwargs["task_name"]
          2. Environment variable MIPS_SCHEDULER_TASK
          3. Default: "easy_alu_chain"
        """
        task_name = kwargs.get(
            "task_name",
            os.environ.get("MIPS_SCHEDULER_TASK", "easy_alu_chain"),
        )
        self._task = get_task(task_name)

        # Parse assembly → instructions
        instructions = parse_assembly(self._task.assembly, self._task.branch_taken_map)

        # Build dependency DAG
        dag = build_dependency_dag(instructions)
        self._dag_dict = dag.to_dict()

        # Initialize tracker (Kahn's algorithm for legal actions)
        self._tracker = ScheduleTracker(dag)

        # Initialize pipeline simulator
        self._simulator = PipelineSimulator()

        # Reset scheduling state
        self._scheduled_instructions = []
        self._schedule_order = []
        self._total_stalls = 0
        self._total_cycles = 0
        self._max_possible_stalls = compute_max_possible_stalls(instructions)

        # Set episode state
        self._state = SchedulerState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_name=task_name,
            difficulty=self._task.difficulty,
            num_instructions=dag.num_nodes,
        )

        return SchedulerObservation(
            done=False,
            reward=0.0,
            num_instructions=dag.num_nodes,
            instructions=self._dag_dict["instructions"],
            edges=self._dag_dict["edges"],
            scheduled_ids=[],
            legal_actions=self._tracker.legal_actions,
            stalls_this_step=0,
            total_stalls=0,
            total_cycles=0,
            task_name=task_name,
            difficulty=self._task.difficulty,
            metadata={
                "task_description": self._task.description,
                "max_steps": self._task.max_steps,
            },
        )

    # ── step ───────────────────────────────────────────────────────────

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> SchedulerObservation:
        """
        Execute a scheduling step: place the chosen instruction next.

        The agent's action must be in the current legal_actions list.
        """
        if not isinstance(action, SchedulerAction):
            raise ValueError(f"Expected SchedulerAction, got {type(action)}")

        if self._tracker is None or self._task is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        inst_id = action.instruction_id

        # Validate action is legal
        legal = self._tracker.legal_actions
        if inst_id not in legal:
            # Return heavily penalized observation without advancing
            return SchedulerObservation(
                done=False,
                reward=-10.0,
                num_instructions=self._state.num_instructions,
                instructions=self._dag_dict["instructions"],
                edges=self._dag_dict["edges"],
                scheduled_ids=list(self._schedule_order),
                legal_actions=legal,
                stalls_this_step=0,
                total_stalls=self._total_stalls,
                total_cycles=self._total_cycles,
                task_name=self._state.task_name,
                difficulty=self._state.difficulty,
                metadata={"error": f"Instruction {inst_id} is not a legal action. Legal: {legal}"},
            )

        # Schedule the instruction
        self._tracker.schedule(inst_id)
        self._schedule_order.append(inst_id)
        inst = self._tracker.dag.instruction_by_id(inst_id)
        self._scheduled_instructions.append(inst)

        # Run pipeline simulation on current schedule to get stalls
        sim = PipelineSimulator()
        result = sim.simulate(self._scheduled_instructions)
        stalls_this_step = result.stall_breakdown[-1] if result.stall_breakdown else 0
        self._total_stalls = result.total_stalls
        self._total_cycles = result.total_cycles

        # Update state
        self._state.step_count += 1

        # Check if done
        done = self._tracker.is_complete
        if self._state.step_count >= self._task.max_steps:
            done = True

        # Compute reward
        step_reward = -stalls_this_step  # per-step penalty for stalls

        if done and self._tracker.is_complete:
            # Episode-end bonus: exp(-total_stalls/max_possible_stalls)
            episode_bonus = compute_episode_reward(
                self._total_stalls, self._max_possible_stalls
            )
            step_reward += episode_bonus

        # Compute final grade if done
        metadata: dict[str, Any] = {}
        if done and self._tracker.is_complete:
            grade = compute_grade(self._state.task_name, self._schedule_order)
            metadata["final_grade"] = grade
            metadata["total_stalls"] = self._total_stalls
            metadata["total_cycles"] = self._total_cycles
            metadata["schedule_order"] = list(self._schedule_order)
        elif done:
            metadata["error"] = "Episode truncated — not all instructions scheduled"
            metadata["scheduled_count"] = len(self._schedule_order)
            metadata["total_count"] = self._state.num_instructions

        return SchedulerObservation(
            done=done,
            reward=round(step_reward, 4),
            num_instructions=self._state.num_instructions,
            instructions=self._dag_dict["instructions"],
            edges=self._dag_dict["edges"],
            scheduled_ids=list(self._schedule_order),
            legal_actions=self._tracker.legal_actions if not done else [],
            stalls_this_step=stalls_this_step,
            total_stalls=self._total_stalls,
            total_cycles=self._total_cycles,
            task_name=self._state.task_name,
            difficulty=self._state.difficulty,
            metadata=metadata,
        )

    # ── state ──────────────────────────────────────────────────────────

    @property
    def state(self) -> SchedulerState:
        """Get the current environment state."""
        return self._state
