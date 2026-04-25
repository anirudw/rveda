# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rveda environment client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import MedicalAction, MedicalObservation
except ImportError:
    from models import MedicalAction, MedicalObservation


class RvedaEnv(
    EnvClient[MedicalAction, MedicalObservation, State]
):
    """
    Client for the Rveda environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with RvedaEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.patient_note)
        ...
        ...     result = client.step(
        ...         MedicalAction(action_type="SEARCH", query="pneumonia")
        ...     )
        ...     print(result.observation.search_results)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = RvedaEnv.from_docker_image("rveda-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(
        ...         MedicalAction(action_type="DETAILS", query="J18.9")
        ...     )
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: MedicalAction) -> Dict:
        """
        Convert MedicalAction to JSON payload for step message.

        Args:
            action: MedicalAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "action_type": action.action_type.value,
            "query": action.query,
        }

    def _parse_result(self, payload: Dict) -> StepResult[MedicalObservation]:
        """
        Parse server response into StepResult[MedicalObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with MedicalObservation
        """
        obs_data = payload.get("observation", {})
        observation = MedicalObservation(
            patient_note=obs_data.get("patient_note", ""),
            search_results=obs_data.get("search_results", []),
            detailed_info=obs_data.get("detailed_info", ""),
            current_reward=obs_data.get("current_reward", 0.0),
            grading=obs_data.get("grading", {}),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
