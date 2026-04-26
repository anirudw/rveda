"""Policy and schema-drift helpers for the Rveda V2 workflow."""

from dataclasses import dataclass, field
from typing import Any, Mapping

try:
    from ..models import (
        ClaimDraftPayload,
        ClaimSchemaState,
        DriftNotice,
        EvidenceSnippet,
        PolicyRule,
        PolicyState,
    )
except ImportError:
    from models import (
        ClaimDraftPayload,
        ClaimSchemaState,
        DriftNotice,
        EvidenceSnippet,
        PolicyRule,
        PolicyState,
    )


@dataclass
class PolicyOperationResult:
    """Structured result returned by policy-engine actions."""

    reward: float = 0.0
    reward_components: dict[str, float] = field(default_factory=dict)
    detailed_info: str = ""
    last_error: str | None = None
    invalid_reason: str | None = None


class PolicyEngine:
    """Manage policy visibility, schema validation, and mid-episode drift."""

    def __init__(self) -> None:
        self.reset(None)

    def reset(self, task: dict[str, Any] | None) -> None:
        self._policy_rules = dict(task.get("policy_rules", {})) if task else {}
        self._policy_checked = False
        configured_schema = str(self._policy_rules.get("active_schema_version", "")).strip()
        fallback_schema = str(task.get("claim_schema_version", "")).strip() if task else ""
        self._active_schema_version = configured_schema or fallback_schema
        claim_schema = dict(self._policy_rules.get("claim_schema", {}))
        claim_schema_version = str(claim_schema.get("version", "")).strip()
        if not claim_schema_version and self._active_schema_version:
            claim_schema["version"] = self._active_schema_version
        self._active_claim_schema = claim_schema
        self._drift_config = dict(task.get("drift", {})) if task else {}
        self._drift_notice: DriftNotice | None = None
        self._drift_triggered = False

    @property
    def drift_notice(self) -> DriftNotice | None:
        return self._drift_notice

    @property
    def active_schema_version(self) -> str:
        return self._active_schema_version

    def policy_state(self) -> PolicyState:
        if not self._policy_checked:
            return PolicyState(checked=False)

        rules = [
            PolicyRule(
                rule_id=str(rule.get("rule_id", "")),
                description=str(rule.get("description", "")),
            )
            for rule in self._policy_rules.get("rules", [])
        ]
        claim_schema = ClaimSchemaState(
            version=str(self._active_claim_schema.get("version", self._active_schema_version)),
            required_fields=list(self._active_claim_schema.get("required_fields", [])),
        )
        return PolicyState(
            checked=True,
            policy_version=str(self._policy_rules.get("version", "")),
            active_schema_version=self._active_schema_version,
            rules=rules,
            claim_schema=claim_schema,
        )

    def maybe_trigger_drift(self, step_count: int) -> None:
        if self._drift_triggered:
            return
        if not self._drift_config.get("enabled"):
            return

        trigger_step = self._drift_config.get("trigger_step")
        if trigger_step is None or step_count < int(trigger_step):
            return

        from_schema_version = self._active_schema_version or str(
            self._drift_config.get("from_schema_version", "")
        )
        to_schema_version = str(self._drift_config.get("to_schema_version", "")).strip()
        if not to_schema_version:
            return

        self._drift_triggered = True
        self._active_schema_version = to_schema_version
        self._active_claim_schema["version"] = to_schema_version
        if "required_fields" in self._drift_config:
            self._active_claim_schema["required_fields"] = list(
                self._drift_config.get("required_fields", [])
            )
        self._drift_notice = DriftNotice(
            active=True,
            trigger_step=int(trigger_step),
            from_schema_version=from_schema_version or None,
            to_schema_version=to_schema_version,
            message=(
                f"Policy/schema drift activated at step {trigger_step}: "
                f"{from_schema_version or 'unknown'} -> {to_schema_version}"
            ),
        )

    def check_policy(self) -> PolicyOperationResult:
        already_checked = self._policy_checked
        self._policy_checked = True

        policy_version = str(self._policy_rules.get("version", ""))
        rule_count = len(self._policy_rules.get("rules", []))
        schema_version = self._active_schema_version
        reward = 0.02 if not already_checked and (policy_version or schema_version or rule_count) else 0.0
        return PolicyOperationResult(
            reward=reward,
            reward_components={
                "base": 0.0,
                "policy_checked": 1.0,
                "policy_repeat": 1.0 if already_checked else 0.0,
                "rule_count": float(rule_count),
                "final": reward,
            },
            detailed_info=(
                f"CHECK_POLICY revealed {policy_version or 'the current policy'} "
                f"with schema {schema_version or 'unknown'} and {rule_count} rule(s)."
            ),
        )

    def validate_claim(
        self,
        payload: ClaimDraftPayload | None,
        revealed_evidence: Mapping[str, EvidenceSnippet],
    ) -> PolicyOperationResult:
        if payload is None:
            return PolicyOperationResult(
                reward=0.0,
                reward_components={"base": 0.0, "invalid_action": 1.0, "final": 0.0},
                detailed_info="VALIDATE_CLAIM_SCHEMA requires a structured payload.",
                last_error="missing_claim_payload",
                invalid_reason="VALIDATE_CLAIM_SCHEMA requires a structured payload.",
            )

        required_fields = list(self._active_claim_schema.get("required_fields", []))
        missing_fields: list[str] = []
        for field_name in required_fields:
            value = getattr(payload, field_name, None)
            if value is None or value == "" or value == []:
                missing_fields.append(field_name)

        if missing_fields:
            message = f"Missing required fields: {', '.join(missing_fields)}"
            return PolicyOperationResult(
                reward=0.0,
                reward_components={
                    "base": 0.0,
                    "missing_required_fields": float(len(missing_fields)),
                    "final": 0.0,
                },
                detailed_info=message,
                last_error="schema_validation_failed",
                invalid_reason=message,
            )

        submitted_schema_version = str(getattr(payload, "schema_version", "")).strip()
        if self._active_schema_version and submitted_schema_version != self._active_schema_version:
            message = (
                "Claim schema version mismatch: "
                f"expected {self._active_schema_version}, got {submitted_schema_version or '<empty>'}"
            )
            return PolicyOperationResult(
                reward=0.0,
                reward_components={"base": 0.0, "schema_version_match": 0.0, "final": 0.0},
                detailed_info=message,
                last_error="schema_version_mismatch",
                invalid_reason=message,
            )

        unknown_evidence_ids = [
            evidence_id for evidence_id in getattr(payload, "evidence_ids", []) if evidence_id not in revealed_evidence
        ]
        if unknown_evidence_ids:
            message = (
                "Draft claim cites evidence that has not been revealed: "
                + ", ".join(unknown_evidence_ids)
            )
            return PolicyOperationResult(
                reward=0.0,
                reward_components={
                    "base": 0.0,
                    "unknown_evidence_ids": float(len(unknown_evidence_ids)),
                    "final": 0.0,
                },
                detailed_info=message,
                last_error="unknown_evidence_ids",
                invalid_reason=message,
            )

        reward = 0.03
        return PolicyOperationResult(
            reward=reward,
            reward_components={
                "base": 0.0,
                "schema_version_match": 1.0,
                "required_fields_present": 1.0,
                "evidence_ids_known": 1.0,
                "final": reward,
            },
            detailed_info=(
                f"VALIDATE_CLAIM_SCHEMA accepted schema {submitted_schema_version or self._active_schema_version} "
                f"with {len(getattr(payload, 'evidence_ids', []))} evidence id(s)."
            ),
        )
