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
from typing import Any

from openenv.core.env_server.types import Action, Observation

from pydantic import BaseModel, Field


class MedicalActionType(str, Enum):
    """Supported agent actions in the medical workflow."""

    SEARCH = "SEARCH"
    DETAILS = "DETAILS"
    SUBMIT = "SUBMIT"
    QUERY_EHR = "QUERY_EHR"


class MedicalAction(Action):
    """Action payload sent by the agent."""

    action_type: MedicalActionType = Field(
        ..., description="Type of medical workflow action to perform"
    )
    query: str = Field(..., description="Search term, code, or submission payload")
    module: str | None = Field(
        default=None,
        description="EHR module key for QUERY_EHR actions",
    )


class SearchResult(BaseModel):
    """Structured search result returned to the agent."""

    code: str = Field(..., description="Medical code identifier")
    short_desc: str = Field(..., description="Short description for the code")


class EvidenceSnippet(BaseModel):
    """Evidence snippet revealed from a hidden EHR module."""

    evidence_id: str = Field(..., description="Stable evidence identifier")
    module: str = Field(..., description="Source EHR module")
    text: str = Field(..., description="Evidence text or structured value")
    supports_codes: list[str] = Field(
        default_factory=list,
        description="ICD codes supported by this evidence",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional evidence fields preserved from the task fixture",
    )


class EhrModuleState(BaseModel):
    """Visible status for a hidden EHR module."""

    status: str = Field(default="closed", description="closed, open, or exhausted")
    query_budget_remaining: int = Field(
        default=0,
        ge=0,
        description="Remaining query budget for the module",
    )
    revealed_count: int = Field(
        default=0,
        ge=0,
        description="Number of snippets revealed from the module",
    )


class GradingTrace(BaseModel):
    """Explicit grading trace for the current episode step."""

    action_type: str = Field(default="", description="Action type that produced this trace")
    grader: str = Field(default="", description="Named grader used for the current task")
    difficulty: str = Field(default="", description="Task difficulty label")
    step: int = Field(default=0, ge=0, description="Environment step count at grading time")
    task_id: str = Field(default="", description="Task identifier")
    reward: float = Field(default=0.0, description="Reward assigned by the grader")
    reward_components: dict[str, float] = Field(
        default_factory=dict,
        description="Named reward components used to compute the final reward",
    )
    search_history: list[str] = Field(
        default_factory=list,
        description="Search queries issued during the episode",
    )
    code_history: list[str] = Field(
        default_factory=list,
        description="DETAILS queries issued during the episode",
    )
    last_search_codes: list[str] = Field(
        default_factory=list,
        description="Codes returned by the most recent SEARCH action",
    )
    excludes1_conflict_seen: bool = Field(
        default=False,
        description="Whether an Excludes1 conflict was detected during the episode",
    )


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
    grading: GradingTrace = Field(
        default_factory=GradingTrace,
        description="Explicit grading trace and episode trajectory signals",
    )
    ehr_map: dict[str, EhrModuleState] = Field(
        default_factory=dict,
        description="Visible map of EHR module statuses",
    )
    revealed_evidence: list[EvidenceSnippet] = Field(
        default_factory=list,
        description="Evidence snippets revealed by QUERY_EHR",
    )
    last_error: str | None = Field(
        default=None,
        description="Machine-readable error category from the previous step",
    )
    invalid_reason: str | None = Field(
        default=None,
        description="Human-readable reason the previous action was invalid",
    )
