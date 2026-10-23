"""Contract for the first Climber task.

This module deliberately contains no robot-specific defaults. It documents the
small interface that the eventual Isaac Lab environment must provide, so the
asset and control decisions stay visible while the model is being prepared.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ClimberInterface:
    """Runtime contract shared by training, sim2sim and deployment."""

    joint_names: Sequence[str]
    observation_names: Sequence[str]
    action_names: Sequence[str]
    policy_frequency_hz: float
    action_scale: Sequence[float]

    def validate(self) -> None:
        if not self.joint_names:
            raise ValueError("joint_names is empty: provide the real actuated joint order")
        if not self.observation_names:
            raise ValueError("observation_names is empty: define the policy observation contract")
        if len(self.action_names) != len(self.action_scale):
            raise ValueError("action_names and action_scale must have identical lengths")
        if self.policy_frequency_hz <= 0:
            raise ValueError("policy_frequency_hz must be positive")


def build_interface() -> ClimberInterface:
    """Return the interface after the robot contract has been filled in."""

    raise NotImplementedError(
        "Fill ClimberInterface from configs/robot/robot.yaml and configs/env/flat.yaml "
        "before creating an Isaac Lab environment."
    )
