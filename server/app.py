"""
FastAPI application for the MIPS Instruction Scheduling Environment.

Usage:
    uvicorn mips_scheduler_env.server.app:app --host 0.0.0.0 --port 8000
"""

from openenv.core.env_server import create_app

from ..models import SchedulerAction, SchedulerObservation
from .mips_scheduler_env import MIPSSchedulerEnvironment

app = create_app(
    MIPSSchedulerEnvironment,
    SchedulerAction,
    SchedulerObservation,
    env_name="mips_scheduler_env",
)


def main():
    """Entry point for direct execution."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
