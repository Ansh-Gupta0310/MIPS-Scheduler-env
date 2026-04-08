"""
Typed Pydantic models for the MIPS Instruction Scheduling environment.
=====================================================================
Defines Action, Observation, and State models following the OpenEnv spec.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openenv.core.env_server.interfaces import Action, Observation, State


class SchedulerAction(Action):
    """
    Agent selects the next instruction to schedule.
    
    The instruction_id must be one of the currently legal actions
    (i.e., all of its dependencies have already been scheduled).
    """
    instruction_id: int


class SchedulerObservation(Observation):
    """
    What the agent observes after each step.

    Contains the full DAG structure (constant throughout the episode),
    the current scheduling state (which instructions have been placed),
    the set of currently legal next moves, and pipeline feedback.
    """
    # ── DAG structure (constant per episode) ──
    num_instructions: int = 0
    instructions: List[Dict[str, Any]] = []
    edges: List[List[Any]] = []

    # ── Current scheduling state ──
    scheduled_ids: List[int] = []
    legal_actions: List[int] = []

    # ── Pipeline feedback ──
    stalls_this_step: int = 0
    total_stalls: int = 0
    total_cycles: int = 0

    # ── Task info ──
    task_name: str = ""
    difficulty: str = ""
    
    # ── Final Grade (Hackathon Requirement) ──
    final_grade: float = 0.0


class SchedulerState(State):
    """
    Episode metadata for the MIPS scheduling environment.
    """
    task_name: str = ""
    difficulty: str = ""
    num_instructions: int = 0
