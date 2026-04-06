# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Rveda environment.

These models define the action and observation payloads used by the medical
coding workflow in the OpenEnv environment.
"""

from enum import Enum

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


class MedicalActionType(str, Enum):
    """Supported agent actions in the medical workflow."""

    SEARCH = "SEARCH"
    DETAILS = "DETAILS"
    SUBMIT = "SUBMIT"


class MedicalAction(Action):
    """Action payload sent by the agent."""

    action_type: MedicalActionType = Field(
        ..., description="Type of medical workflow action to perform"
    )
    query: str = Field(..., description="Search term, code, or submission payload")


class SearchResult(BaseModel):
    """Structured search result returned to the agent."""

    code: str = Field(..., description="Medical code identifier")
    short_desc: str = Field(..., description="Short description for the code")


class MedicalObservation(Observation):
    """Observation payload returned by the environment."""

    patient_note: str = Field(default="", description="Current patient note context")
    search_results: list[SearchResult] = Field(
        default_factory=list,
        description="Candidate search results with code identifiers and short descriptions",
    )
    detailed_info: str = Field(
        default="", description="Detailed information for the selected medical code"
    )
    current_reward: float = Field(
        default=0.0, description="Current reward accumulated in the episode"
    )
