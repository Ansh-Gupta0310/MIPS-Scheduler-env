"""
MIPS Instruction Scheduling Environment for OpenEnv.

An RL environment where agents learn to schedule MIPS assembly instructions
to minimize pipeline stalls in a simulated 5-stage pipelined processor.

Example:
    >>> import asyncio
    >>> from mips_scheduler_env import MIPSSchedulerEnv, SchedulerAction
    >>>
    >>> async def main():
    ...     env = await MIPSSchedulerEnv.from_docker_image("mips-scheduler-env:latest")
    ...     async with env:
    ...         result = await env.reset(task_name="easy_alu_chain")
    ...         while not result.done:
    ...             action = SchedulerAction(instruction_id=result.observation.legal_actions[0])
    ...             result = await env.step(action)
    ...         print(f"Score: {result.observation.metadata.get('final_grade')}")
    >>>
    >>> asyncio.run(main())
"""

from .client import MIPSSchedulerEnv
from .models import SchedulerAction, SchedulerObservation, SchedulerState

__all__ = [
    "MIPSSchedulerEnv",
    "SchedulerAction",
    "SchedulerObservation",
    "SchedulerState",
]
