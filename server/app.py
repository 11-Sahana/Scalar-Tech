"""
server/app.py
-------------
FastAPI application entry point for the Supply Disruption Responder.
Uses openenv.core.env_server.create_fastapi_app to expose the
standard OpenEnv HTTP interface: /reset, /step, /state, /health
"""

from __future__ import annotations

try:
    from openenv.core.env_server import create_fastapi_app
except ImportError:
    raise RuntimeError(
        "openenv-core is required to run the server.\n"
        "Install it with: pip install openenv-core"
    )

try:
    from ..app.models import Action, Observation
    from .supply_disruption_environment import SupplyDisruptionServerEnvironment
except ImportError:
    from app.models import Action, Observation
    from server.supply_disruption_environment import SupplyDisruptionServerEnvironment

# create_fastapi_app accepts a class (not an instance)
app = create_fastapi_app(
    SupplyDisruptionServerEnvironment,
    Action,
    Observation,
    env_name="supply-disruption-env",
)


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
