"""Gym registration will be enabled when the first Climber task is runnable."""


def register_tasks() -> None:
    """Register Climber-Flat-v0 after env_cfg and env are implemented."""

    raise NotImplementedError(
        "Do not register a task until the Climber asset, observation/action spaces, "
        "reset logic, rewards and termination conditions are defined."
    )
