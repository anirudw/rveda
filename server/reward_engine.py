"""Reward rubric helpers for the Rveda V2 workflow."""

from dataclasses import dataclass
from typing import Any, Mapping

try:
    from ..models import DriftNotice, EvidenceSnippet, RewardMetrics, ReasoningLog
    from .policy_engine import PolicyEngine
except ImportError:
    from models import DriftNotice, EvidenceSnippet, RewardMetrics, ReasoningLog
    from server.policy_engine import PolicyEngine


@dataclass
class RewardEvaluation:
    """Structured result returned by the reward engine."""

    reward: float
    reward_metrics: RewardMetrics


class RewardEngine:
    """Compute rubric-level metrics and final capped reward."""

    def __init__(self) -> None:
        self.reset(None)

    def reset(self, task: dict[str, Any] | None) -> None:
        self._task = task or {}

    def evaluate(
        self,
        *,
        action_type: str,
        base_reward: float,
        reward_components: Mapping[str, float],
        step_count: int,
        search_history: list[str],
        code_history: list[str],
        last_error: str | None,
        invalid_reason: str | None,
        reasoning_log: ReasoningLog | None,
        reasoning_log_verified: bool,
        schema_validation_passed: bool,
        validated_schema_version: str | None,
        drift_notice: DriftNotice | None,
        policy_engine: PolicyEngine,
        timed_out: bool,
    ) -> RewardEvaluation:
        terminal_correctness = self._terminal_correctness(action_type, reward_components)
        evidence_grounding = self._evidence_grounding(
            action_type=action_type,
            reasoning_log=reasoning_log,
            reasoning_log_verified=reasoning_log_verified,
        )
        schema_compliance = self._schema_compliance(
            action_type=action_type,
            base_reward=base_reward,
            last_error=last_error,
            reasoning_log_verified=reasoning_log_verified,
        )
        format_validity = self._format_validity(
            action_type=action_type,
            last_error=last_error,
            invalid_reason=invalid_reason,
        )
        process_discipline = self._process_discipline(
            action_type=action_type,
            step_count=step_count,
            search_history=search_history,
            code_history=code_history,
            last_error=last_error,
            timed_out=timed_out,
        )
        drift_adaptation = self._drift_adaptation(
            action_type=action_type,
            policy_engine=policy_engine,
            drift_notice=drift_notice,
            reasoning_log_verified=reasoning_log_verified,
            schema_validation_passed=schema_validation_passed,
            validated_schema_version=validated_schema_version,
            submit_valid=action_type == "SUBMIT" and last_error is None and base_reward > 0.0,
        )

        final = self._combine_reward(
            action_type=action_type,
            base_reward=base_reward,
            terminal_correctness=terminal_correctness,
            evidence_grounding=evidence_grounding,
            schema_compliance=schema_compliance,
            format_validity=format_validity,
            process_discipline=process_discipline,
            drift_adaptation=drift_adaptation,
            timed_out=timed_out,
        )

        return RewardEvaluation(
            reward=final,
            reward_metrics=RewardMetrics(
                terminal_correctness=terminal_correctness,
                evidence_grounding=evidence_grounding,
                schema_compliance=schema_compliance,
                format_validity=format_validity,
                process_discipline=process_discipline,
                drift_adaptation=drift_adaptation,
                final=final,
            ),
        )

    def _terminal_correctness(self, action_type: str, reward_components: Mapping[str, float]) -> float:
        if action_type != "SUBMIT":
            return 0.0
        if reward_components.get("exact_match", 0.0) >= 1.0:
            return 1.0
        if reward_components.get("family_match", 0.0) >= 1.0:
            return 0.55
        return 0.0

    def _evidence_grounding(
        self,
        *,
        action_type: str,
        reasoning_log: ReasoningLog | None,
        reasoning_log_verified: bool,
    ) -> float:
        if reasoning_log_verified and reasoning_log is not None:
            return 1.0
        if action_type in {"VALIDATE_CLAIM_SCHEMA", "REASONING_LOG"}:
            return 0.45
        return 0.0

    def _schema_compliance(
        self,
        *,
        action_type: str,
        base_reward: float,
        last_error: str | None,
        reasoning_log_verified: bool,
    ) -> float:
        if action_type == "CHECK_POLICY":
            return 1.0 if last_error is None else 0.0
        if action_type == "VALIDATE_CLAIM_SCHEMA":
            return 1.0 if last_error is None and base_reward > 0.0 else 0.0
        if action_type == "REASONING_LOG":
            return 1.0 if reasoning_log_verified else 0.0
        if action_type == "SUBMIT":
            return 1.0 if last_error is None and base_reward > 0.0 else 0.0
        return 0.0

    def _format_validity(
        self,
        *,
        action_type: str,
        last_error: str | None,
        invalid_reason: str | None,
    ) -> float:
        if action_type in {"QUERY_EHR", "CHECK_POLICY", "VALIDATE_CLAIM_SCHEMA", "REASONING_LOG", "SUBMIT"}:
            return 1.0 if last_error is None and invalid_reason is None else 0.0
        return 1.0

    def _process_discipline(
        self,
        *,
        action_type: str,
        step_count: int,
        search_history: list[str],
        code_history: list[str],
        last_error: str | None,
        timed_out: bool,
    ) -> float:
        if timed_out:
            return 0.0

        score = 1.0
        if last_error is not None:
            score -= 0.35

        if len(search_history) != len(set(search_history)):
            score -= 0.2
        if len(code_history) != len(set(code_history)):
            score -= 0.2
        if step_count > 6:
            score -= 0.05 * (step_count - 6)
        if action_type in {"CHECK_POLICY", "VALIDATE_CLAIM_SCHEMA", "REASONING_LOG"}:
            score += 0.05

        return max(0.0, min(score, 1.0))

    def _drift_adaptation(
        self,
        *,
        action_type: str,
        policy_engine: PolicyEngine,
        drift_notice: DriftNotice | None,
        reasoning_log_verified: bool,
        schema_validation_passed: bool,
        validated_schema_version: str | None,
        submit_valid: bool,
    ) -> float:
        if drift_notice is None or not drift_notice.active:
            return 1.0

        active_schema = policy_engine.active_schema_version
        drift_schema = drift_notice.to_schema_version
        adapted_schema = (
            schema_validation_passed
            and validated_schema_version is not None
            and validated_schema_version == active_schema
            and active_schema == drift_schema
        )
        if action_type == "VALIDATE_CLAIM_SCHEMA":
            return 0.75 if adapted_schema else 0.0
        if action_type == "SUBMIT":
            return 1.0 if adapted_schema and submit_valid and reasoning_log_verified else 0.0
        return 0.0

    def _combine_reward(
        self,
        *,
        action_type: str,
        base_reward: float,
        terminal_correctness: float,
        evidence_grounding: float,
        schema_compliance: float,
        format_validity: float,
        process_discipline: float,
        drift_adaptation: float,
        timed_out: bool,
    ) -> float:
        if timed_out:
            return 0.0
        if action_type == "SUBMIT":
            combined = (
                0.72 * terminal_correctness
                + 0.12 * evidence_grounding
                + 0.08 * schema_compliance
                + 0.04 * format_validity
                + 0.03 * process_discipline
                + 0.01 * drift_adaptation
            )
            return max(0.0, min(combined, 0.99))

        combined = max(
            base_reward,
            min(
                0.05,
                base_reward
                + 0.01 * evidence_grounding
                + 0.01 * schema_compliance
                + 0.01 * format_validity
                + 0.01 * process_discipline
                + 0.005 * drift_adaptation,
            ),
        )
        return max(0.0, min(combined, 0.05))
