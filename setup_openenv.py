"""Run this script to generate pyproject.toml and server/ correctly."""
import os

# Write pyproject.toml
pyproject = """[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "supply-disruption-env"
version = "1.0.0"
description = "OpenEnv supply chain disruption RL environment"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.0,<3.0",
    "openai>=1.30,<2.0",
    "anthropic>=0.25,<1.0",
    "pyyaml>=6.0",
    "fastapi",
    "uvicorn",
    "openenv-core>=0.2.0",
]

[project.scripts]
server = "server.app:main"

[tool.setuptools.packages.find]
where = ["."]
include = ["app*", "agent*", "server*"]
"""

with open("pyproject.toml", "w", encoding="utf-8") as f:
    f.write(pyproject)
print("✓ pyproject.toml written")

# Write server/__init__.py
os.makedirs("server", exist_ok=True)
with open("server/__init__.py", "w", encoding="utf-8") as f:
    f.write("")
print("✓ server/__init__.py written")

# Write server/app.py
server_app = '''"""
server/app.py
-------------
FastAPI server exposing the Supply Disruption environment
as an HTTP API compatible with the OpenEnv spec.

Runs on port 7860 (required by Hugging Face Spaces).
"""
from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from fastapi import FastAPI
from app.env import SupplyDisruptionEnv
from app.models import Action

app = FastAPI(title="Supply Disruption Responder", version="1.0.0")

# Shared environment instance
env = SupplyDisruptionEnv(seed=42)
obs = env.reset()


@app.get("/")
def root():
    return {"name": "supply-disruption-responder", "version": "1.0.0", "status": "ok"}


@app.post("/reset")
def reset():
    global obs
    obs = env.reset()
    return obs.dict()


@app.post("/step")
def step(action: Action):
    global obs
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.dict(),
        "reward": reward.dict(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state():
    return env.state()


@app.get("/health")
def health():
    return {"status": "ok"}


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
'''

with open("server/app.py", "w", encoding="utf-8") as f:
    f.write(server_app)
print("✓ server/app.py written")
print("\nAll done! Now run:  uv lock  then  openenv validate .")
