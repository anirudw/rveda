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

import json

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import MedicalAction, MedicalObservation
    from .rveda_environment import InvalidTaskIdError, RvedaEnvironment
except ImportError:
    from models import MedicalAction, MedicalObservation
    from server.rveda_environment import InvalidTaskIdError, RvedaEnvironment

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


SENSITIVE_REWARD_COMPONENT_KEYS = {
    "target_hit_bonus",
    "family_hit_bonus",
    "exact_match",
    "family_match",
    "family_bonus",
    "detail_relevant",
}


def _sanitize_visible_grading(grading: dict) -> dict:
    if not isinstance(grading, dict):
        return {}
    sanitized = dict(grading)
    reward_components = sanitized.get("reward_components")
    if isinstance(reward_components, dict):
        sanitized["reward_components"] = {
            key: value
            for key, value in reward_components.items()
            if key not in SENSITIVE_REWARD_COMPONENT_KEYS
        }
    return sanitized


# Create the app with web interface and README integration
app = create_app(
    RvedaEnvironment,
    MedicalAction,
    MedicalObservation,
    env_name="rveda",
    max_concurrent_envs=2,  # conservative default: allow a small number of concurrent WebSocket sessions
)


class StepInfoMiddleware(BaseHTTPMiddleware):
    """Ensure /step responses include the grader-required info field."""

    async def dispatch(self, request, call_next):
        response = await call_next(request)
        if request.url.path != "/step" or response.status_code != 200:
            return response

        content_type = response.headers.get("content-type", "")
        if "application/json" not in content_type:
            return response

        body = b"".join([chunk async for chunk in response.body_iterator])
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return JSONResponse(content={"detail": "Invalid JSON response from /step"}, status_code=500)

        if isinstance(payload, dict):
            grading = payload.get("observation", {}).get("grading", {})
            payload["info"] = _sanitize_visible_grading(grading)

        return JSONResponse(content=payload, status_code=response.status_code, headers=dict(response.headers))


app.add_middleware(StepInfoMiddleware)


@app.exception_handler(InvalidTaskIdError)
async def invalid_task_id_handler(_request: Request, exc: InvalidTaskIdError):
    return JSONResponse(
        status_code=422,
        content={
            "detail": [
                {
                    "type": "value_error.invalid_task_id",
                    "loc": ["body", "task_id"],
                    "msg": str(exc),
                    "input": exc.task_id,
                }
            ]
        },
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
