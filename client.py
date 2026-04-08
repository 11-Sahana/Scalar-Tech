"""
client.py
---------
OpenEnv client for the Supply Disruption Responder environment.
Provides a local client interface to interact with the environment.
"""

from __future__ import annotations
from typing import Any

try:
    from openenv.core.env_client import EnvironmentClient
except ImportError:
    # Fallback for local dev
    class EnvironmentClient:
        pass

try:
    from app.env import SupplyDisruptionEnv
    from app.models import Action, Observation
except ImportError:
    from .app.env import SupplyDisruptionEnv
    from .app.models import Action, Observation


class SupplyDisruptionClient(EnvironmentClient):
    """
    OpenEnv-compatible client wrapper around SupplyDisruptionEnv.
    Allows local interaction with the environment.
    """

    def __init__(self):
        super().__init__()
        self._env = SupplyDisruptionEnv(seed=42)

    def reset(self) -> Observation:
        return self._env.reset()

    def step(self, action: Action) -> tuple[Observation, Any, bool, dict]:
        return self._env.step(action)

    def state(self) -> dict:
        return self._env.state()</content>
<parameter name="filePath">c:\supply-disruption-env\supply-disruption-env\client.py