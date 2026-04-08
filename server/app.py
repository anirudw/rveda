# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Rveda Environment.

This module creates an HTTP server that exposes the RvedaEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from models import MedicalAction, MedicalObservation
    from .rveda_environment import RvedaEnvironment
except ModuleNotFoundError:
    from models import MedicalAction, MedicalObservation
    from server.rveda_environment import RvedaEnvironment


# Create the app with web interface and README integration
app = create_app(
    RvedaEnvironment,
    MedicalAction,
    MedicalObservation,
    env_name="rveda",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)
@app.get("/health")
async def health():
    return {"status": "ok"}
def main():
    """
    Entry point for direct execution.
    """
    import uvicorn
    import os
    
    # Prioritize the HF environment variable, then the default 8000
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0"
    
    print(f"Starting Rveda server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()