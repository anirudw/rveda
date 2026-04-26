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
    QUERY_EHR = "QUERY_EHR"
    CHECK_POLICY = "CHECK_POLICY"
    VALIDATE_CLAIM_SCHEMA = "VALIDATE_CLAIM_SCHEMA"
    REASONING_LOG = "REASONING_LOG"
    SUBMIT = "SUBMIT"


class MedicalAction(Action):
    """Action payload sent by the agent."""

    action_type: MedicalActionType = Field(
        ..., description="Type of medical workflow action to perform"
    )
    query: str = Field(
        default="",
        description="Search term, code, or free-form submission payload",
    )
    module: str | None = Field(
        default=None,
        description="EHR module key for QUERY_EHR actions",
    )
    payload: "ClaimDraftPayload | ReasoningLogPayload | None" = Field(
        default=None,
        description="Structured payload for VALIDATE_CLAIM_SCHEMA or REASONING_LOG",
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


class ClaimDraftPayload(BaseModel):
    """Draft claim payload used for schema validation."""

    diagnosis_code: str = Field(default="", description="Diagnosis code to validate")
    schema_version: str = Field(default="", description="Claim schema version supplied by the agent")
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="Evidence identifiers cited by the draft claim",
    )
    policy_attestations: list[str] = Field(
        default_factory=list,
        description="Policy rule identifiers or attestations included in the draft claim",
    )
    reasoning_log_id: str | None = Field(
        default=None,
        description="Optional reasoning log reference; required by later schemas when configured",
    )


class ReasoningLogPayload(BaseModel):
    """Structured payload for a grounded reasoning record."""

    candidate_code: str = Field(default="", description="Candidate diagnosis code being justified")
    rationale: str = Field(default="", description="Grounded explanation for the candidate code")
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="Revealed evidence identifiers cited by the reasoning record",
    )
    policy_rule_ids: list[str] = Field(
        default_factory=list,
        description="Checked policy rules cited by the reasoning record",
    )


class ReasoningLog(BaseModel):
    """Latest reasoning record stored for the current episode."""

    reasoning_log_id: str = Field(default="", description="Stable identifier for the reasoning record")
    candidate_code: str = Field(default="", description="Candidate diagnosis code being justified")
    rationale: str = Field(default="", description="Grounded explanation text")
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="Evidence identifiers cited by the reasoning record",
    )
    policy_rule_ids: list[str] = Field(
        default_factory=list,
        description="Policy rule identifiers cited by the reasoning record",
    )
    verified: bool = Field(default=False, description="Whether the reasoning record passed verification")


class PolicyRule(BaseModel):
    """Visible policy rule exposed after CHECK_POLICY."""

    rule_id: str = Field(default="", description="Stable policy rule identifier")
    description: str = Field(default="", description="Human-readable policy rule text")


class ClaimSchemaState(BaseModel):
    """Visible claim schema metadata surfaced to the agent."""

    version: str = Field(default="", description="Active claim schema version")
    required_fields: list[str] = Field(
        default_factory=list,
        description="Required fields for the active schema",
    )


class PolicyState(BaseModel):
    """Visible policy and schema state for the current episode."""

    checked: bool = Field(default=False, description="Whether CHECK_POLICY has been called")
    policy_version: str = Field(default="", description="Active policy version")
    active_schema_version: str = Field(default="", description="Currently active claim schema version")
    rules: list[PolicyRule] = Field(
        default_factory=list,
        description="Policy rules visible to the agent",
    )
    claim_schema: ClaimSchemaState = Field(
        default_factory=ClaimSchemaState,
        description="Current claim schema metadata",
    )


class DriftNotice(BaseModel):
    """Structured notice emitted when policy/schema drift occurs mid-episode."""

    active: bool = Field(default=False, description="Whether a drift event is currently being surfaced")
    trigger_step: int | None = Field(
        default=None,
        ge=0,
        description="Step count at which the drift was triggered",
    )
    from_schema_version: str | None = Field(
        default=None,
        description="Schema version active before drift",
    )
    to_schema_version: str | None = Field(
        default=None,
        description="Schema version active after drift",
    )
    message: str = Field(default="", description="Human-readable drift summary")


class RewardMetrics(BaseModel):
    """Rubric-level reward metrics exposed for debugging and training."""

    terminal_correctness: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_grounding: float = Field(default=0.0, ge=0.0, le=1.0)
    schema_compliance: float = Field(default=0.0, ge=0.0, le=1.0)
    format_validity: float = Field(default=0.0, ge=0.0, le=1.0)
    process_discipline: float = Field(default=0.0, ge=0.0, le=1.0)
    drift_adaptation: float = Field(default=0.0, ge=0.0, le=1.0)
    final: float = Field(default=0.0, ge=0.0, le=1.0)


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
    policy_state: PolicyState = Field(
        default_factory=PolicyState,
        description="Visible insurance policy and schema state",
    )
    drift_notice: DriftNotice | None = Field(
        default=None,
        description="Structured notice when policy/schema drift occurs",
    )
    reasoning_log: ReasoningLog | None = Field(
        default=None,
        description="Latest submitted reasoning record for the episode",
    )
    reasoning_log_verified: bool = Field(
        default=False,
        description="Whether the latest reasoning record passed verification",
    )
    reward_metrics: RewardMetrics = Field(
        default_factory=RewardMetrics,
        description="Rubric-level numeric reward metrics",
    )
    last_error: str | None = Field(
        default=None,
        description="Machine-readable error category from the previous step",
    )
    invalid_reason: str | None = Field(
        default=None,
        description="Human-readable reason the previous action was invalid",
    )
