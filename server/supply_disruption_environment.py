"""
server/supply_disruption_environment.py
----------------------------------------
OpenEnv server-side Environment wrapper.
Wraps the core SupplyDisruptionEnv to conform to the
openenv.core.env_server.Environment interface.
"""

from __future__ import annotations
import uuid
from typing import Any

try:
    from openenv.core.env_server import Environment  # type: ignore
except ImportError:
    # Fallback base class when openenv-core is not installed (local dev)
    from typing import Generic, TypeVar
    ActT = TypeVar('ActT')
    ObsT = TypeVar('ObsT')
    StateT = TypeVar('StateT')
    class Environment(Generic[ActT, ObsT, StateT]):
        pass

try:
    from ..app.env import SupplyDisruptionEnv
    from ..app.models import Action, Observation
except ImportError:
    from app.env import SupplyDisruptionEnv
    from app.models import Action, Observation


class SupplyDisruptionServerEnvironment(Environment):
    """
    OpenEnv-compatible server wrapper around SupplyDisruptionEnv.
    Handles one episode per session.
    """

    def __init__(self):
        super().__init__()
        self._env = SupplyDisruptionEnv(seed=42)
        self._obs = None
        self._done = False
        self._episode_id = None

    def reset(self) -> Observation:
        self._episode_id = str(uuid.uuid4())
        self._done = False
        self._obs = self._env.reset()
        return self._obs

    def step(self, action: Action) -> Observation:
        if self._done:
            assert self._obs is not None
            return self._obs

        self._obs, reward, self._done, info = self._env.step(action)
        # Attach reward info to observation metrics for client visibility
        self._obs.metrics["last_reward"] = reward.value
        self._obs.metrics["reward_breakdown"] = reward.breakdown.dict()
        self._obs.metrics["step_info"] = info
        return self._obs

    @property
    def state(self) -> dict:
        return self._env.state()
