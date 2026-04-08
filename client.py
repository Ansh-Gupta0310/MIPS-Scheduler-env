"""
MIPS Scheduler Environment Client
-----------------------------------
Client-side wrapper for connecting to the MIPS Scheduler environment server.

Uses EnvClient (WebSocket-based persistent connection) for efficient
multi-step interactions during scheduling episodes.
"""

from __future__ import annotations

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import SchedulerAction, SchedulerObservation, SchedulerState


class MIPSSchedulerEnv(EnvClient[SchedulerAction, SchedulerObservation, SchedulerState]):
    """
    Client for the MIPS Instruction Scheduling Environment.

    Example (async):
        >>> async with MIPSSchedulerEnv(base_url="http://localhost:8000") as env:
        ...     result = await env.reset(task_name="easy_alu_chain")
        ...     while not result.done:
        ...         inst_id = result.observation.legal_actions[0]
        ...         result = await env.step(SchedulerAction(instruction_id=inst_id))
        ...     print(f"Grade: {result.observation.metadata['final_grade']}")

    Example (sync):
        >>> with MIPSSchedulerEnv(base_url="http://localhost:8000").sync() as env:
        ...     result = env.reset(task_name="easy_alu_chain")
        ...     # ... step loop ...
    """

    def _step_payload(self, action: SchedulerAction) -> Dict[str, Any]:
        """Convert action to JSON payload for the server."""
        return {"instruction_id": action.instruction_id}

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[SchedulerObservation]:
        """Parse server response into StepResult."""
        obs_data = payload.get("observation", payload)
        obs = SchedulerObservation(**obs_data)
        return StepResult(
            observation=obs,
            reward=payload.get("reward", obs.reward),
            done=payload.get("done", obs.done),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> SchedulerState:
        """Parse server state response."""
        return SchedulerState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_name=payload.get("task_name", ""),
            difficulty=payload.get("difficulty", ""),
            num_instructions=payload.get("num_instructions", 0),
        )
