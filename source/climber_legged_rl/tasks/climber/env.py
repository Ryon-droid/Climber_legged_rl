"""Isaac Lab environment entry point reserved for the Climber robot."""

from __future__ import annotations


class ClimberLeggedEnv:
    """Placeholder until the asset and Isaac Lab version are fixed.

    The reference repositories use different simulator generations and APIs.
    Keeping this class unimplemented prevents accidentally training against a
    reference robot with the wrong dynamics.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "ClimberLeggedEnv is not implemented yet. Provide the Climber USD/URDF, "
            "Isaac Lab version, joint contract and control interface first."
        )
