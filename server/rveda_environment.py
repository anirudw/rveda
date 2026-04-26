# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Rveda environment implementation.

This environment exposes a task-driven medical-coding workflow backed by the
local ICD-10 SQLite engine.
"""

import json
import random
from pathlib import Path
from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        EhrModuleState,
        EvidenceSnippet,
        GradingTrace,
        MedicalAction,
        MedicalActionType,
        MedicalObservation,
        ReasoningLog,
        ReasoningLogPayload,
        RewardMetrics,
        SearchResult,
    )
    from .engine import get_code_details, initialize_db, search_codes
    from .policy_engine import PolicyEngine
    from .reward_engine import RewardEngine
except ImportError:
    from models import (
        EhrModuleState,
        EvidenceSnippet,
        GradingTrace,
        MedicalAction,
        MedicalActionType,
        MedicalObservation,
        ReasoningLog,
        ReasoningLogPayload,
        RewardMetrics,
        SearchResult,
    )
    from server.engine import get_code_details, initialize_db, search_codes
    from server.policy_engine import PolicyEngine
    from server.reward_engine import RewardEngine


class InvalidTaskIdError(ValueError):
    """Raised when reset is called with an unknown task identifier."""

    def __init__(self, task_id: str):
        super().__init__(f"Unknown task_id: {task_id}")
        self.task_id = task_id


class RvedaEnvironment(Environment):
    """
    A task-driven medical-coding environment.
    """

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    _MAX_EPISODE_STEPS = 8
    _MIN_OPEN_SCORE = 0.0
    _MAX_OPEN_SCORE = 0.99
    _BASE_REWARD = {
        "exact": 0.72,
        "family": 0.46,
        "wrong": 0.12,
    }
    _SENSITIVE_REWARD_COMPONENT_KEYS = {
        "target_hit_bonus",
        "family_hit_bonus",
        "exact_match",
        "family_match",
        "family_bonus",
        "detail_relevant",
    }
    _GRADER_POLICIES = {
        "easy": {
            "search_novelty_bonus": 0.01,
            "search_result_bonus": 0.01,
            "search_target_bonus": 0.03,
            "search_family_bonus": 0.02,
            "detail_bonus": 0.01,
            "detail_search_hit_bonus": 0.02,
            "detail_family_bonus": 0.01,
            "detail_conflict_penalty": 0.02,
            "submit_search_history_bonus": 0.01,
            "submit_search_hit_bonus": 0.02,
            "submit_code_history_bonus": 0.01,
            "submit_depth_bonus": 0.00,
            "submit_conflict_penalty": 0.03,
        },
        "medium": {
            "search_novelty_bonus": 0.015,
            "search_result_bonus": 0.015,
            "search_target_bonus": 0.04,
            "search_family_bonus": 0.025,
            "detail_bonus": 0.015,
            "detail_search_hit_bonus": 0.03,
            "detail_family_bonus": 0.015,
            "detail_conflict_penalty": 0.04,
            "submit_search_history_bonus": 0.02,
            "submit_search_hit_bonus": 0.03,
            "submit_code_history_bonus": 0.02,
            "submit_depth_bonus": 0.01,
            "submit_conflict_penalty": 0.12,
        },
        "hard": {
            "search_novelty_bonus": 0.02,
            "search_result_bonus": 0.02,
            "search_target_bonus": 0.05,
            "search_family_bonus": 0.03,
            "detail_bonus": 0.02,
            "detail_search_hit_bonus": 0.04,
            "detail_family_bonus": 0.02,
            "detail_conflict_penalty": 0.05,
            "submit_search_history_bonus": 0.03,
            "submit_search_hit_bonus": 0.04,
            "submit_code_history_bonus": 0.03,
            "submit_depth_bonus": 0.02,
            "submit_conflict_penalty": 0.15,
        },
    }

    def __init__(self):
        """Initialize the Rveda environment."""
        initialize_db()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._tasks = self._load_tasks()
        self._v2_tasks = self._load_v2_tasks()
        self._all_tasks = [*self._tasks, *self._v2_tasks]
        self._current_task: dict[str, Any] | None = None
        self._patient_note = ""
        self._search_results: list[SearchResult] = []
        self._detailed_info = ""
        self.code_history: list[str] = []
        self.search_history: list[str] = []
        self._search_result_history: set[tuple[str, ...]] = set()
        self._last_search_codes: set[str] = set()
        self._excludes1_conflict_seen = False
        self._ehr_modules: dict[str, dict[str, Any]] = {}
        self._ehr_remaining_budget: dict[str, int] = {}
        self._revealed_evidence: dict[str, EvidenceSnippet] = {}
        self._last_error: str | None = None
        self._invalid_reason: str | None = None
        self._reasoning_log: ReasoningLog | None = None
        self._reasoning_log_verified = False
        self._schema_validation_passed = False
        self._last_validated_schema_version: str | None = None
        self._policy_engine = PolicyEngine()
        self._reward_engine = RewardEngine()
        self._reward_metrics: RewardMetrics | None = None
        self._episode_done = False

    def _grade_bounds(self, score: float) -> float:
        """Clamp score into the supported reward range."""
        return min(max(score, self._MIN_OPEN_SCORE), self._MAX_OPEN_SCORE)

    def _task_difficulty(self) -> str:
        if not self._current_task:
            return "medium"
        return self._current_task.get("difficulty", "medium").lower()

    def _grader_name(self) -> str:
        return f"{self._task_difficulty()}_grader"

    def _policy(self) -> dict[str, float]:
        return self._GRADER_POLICIES.get(self._task_difficulty(), self._GRADER_POLICIES["medium"])

    def _target_code(self) -> str:
        return self._current_task.get("target_code", "") if self._current_task else ""

    def _same_family(self, submitted_code: str, target_code: str) -> bool:
        return bool(submitted_code and target_code and submitted_code[:3] == target_code[:3])

    def _grade_search_step(
        self,
        query: str,
        result_codes: list[str],
        target_code: str,
    ) -> tuple[float, dict[str, float]]:
        policy = self._policy()
        result_count = len(result_codes)
        result_signature = tuple(result_codes)
        query_repeated = query in self.search_history[:-1]
        result_repeated = bool(result_signature) and result_signature in self._search_result_history
        exact_hit = 1.0 if target_code in result_codes else 0.0
        family_hit = 1.0 if any(self._same_family(code, target_code) for code in result_codes) else 0.0
        is_novel_search = result_count > 0 and not query_repeated and not result_repeated
        novelty_bonus = policy["search_novelty_bonus"] if is_novel_search else 0.0
        result_count_bonus = min(result_count, 5) * policy["search_result_bonus"] if is_novel_search else 0.0
        target_hit_bonus = exact_hit * policy["search_target_bonus"] if is_novel_search else 0.0
        family_hit_bonus = family_hit * policy["search_family_bonus"] if is_novel_search else 0.0
        score = self._grade_bounds(
            (
                0.0
                + novelty_bonus
                + result_count_bonus
                + target_hit_bonus
                + family_hit_bonus
            )
        )
        return score, {
            "base": 0.0,
            "query_repeated": 1.0 if query_repeated else 0.0,
            "result_repeated": 1.0 if result_repeated else 0.0,
            "query_novelty_bonus": novelty_bonus,
            "result_count_bonus": result_count_bonus,
            "target_hit_bonus": target_hit_bonus,
            "family_hit_bonus": family_hit_bonus,
            "result_count": float(result_count),
            "final": score,
        }

    def _grade_details_step(
        self,
        query: str,
        details: dict[str, str] | None,
        target_code: str,
        conflict_seen: bool,
    ) -> tuple[float, dict[str, float]]:
        policy = self._policy()
        detail_seen_before = 1.0 if query in self.code_history else 0.0
        detail_novel = 1.0 - detail_seen_before
        detail_relevant = 1.0 if (query in self._last_search_codes or self._same_family(query, target_code) or query == target_code) else 0.0
        search_hit = 1.0 if query in self._last_search_codes else 0.0
        family_hit = 1.0 if self._same_family(query, target_code) else 0.0
        conflict_penalty = -policy["detail_conflict_penalty"] if conflict_seen else 0.0
        detail_bonus = policy["detail_bonus"] if details and detail_relevant and detail_novel else 0.0
        score = self._grade_bounds(
            (
                0.0
                + detail_bonus
                + search_hit * policy["detail_search_hit_bonus"]
                + family_hit * policy["detail_family_bonus"]
                + conflict_penalty
            )
        )
        return score, {
            "base": 0.0,
            "detail_relevant": detail_relevant,
            "detail_novel": detail_novel,
            "detail_found_bonus": detail_bonus,
            "search_alignment_bonus": search_hit * policy["detail_search_hit_bonus"],
            "family_bonus": family_hit * policy["detail_family_bonus"],
            "conflict_penalty": conflict_penalty,
            "final": score,
        }

    def _grade_submission(
        self,
        submitted_code: str,
        target_code: str,
    ) -> tuple[float, str, dict[str, float]]:
        policy = self._policy()
        grader_name = self._grader_name()

        if not target_code:
            score = self._grade_bounds(self._BASE_REWARD["wrong"])
            return score, grader_name, {"base": score, "final": score}

        if submitted_code == target_code:
            base = self._BASE_REWARD["exact"]
            exact_match = 1.0
            family_match = 0.0
        elif self._same_family(submitted_code, target_code):
            base = self._BASE_REWARD["family"]
            exact_match = 0.0
            family_match = 1.0
        else:
            base = self._BASE_REWARD["wrong"]
            exact_match = 0.0
            family_match = 0.0

        search_history_bonus = policy["submit_search_history_bonus"] if self.search_history else 0.0
        search_hit_bonus = policy["submit_search_hit_bonus"] if submitted_code in self._last_search_codes else 0.0
        code_history_bonus = policy["submit_code_history_bonus"] if submitted_code in self.code_history else 0.0
        depth_bonus = policy["submit_depth_bonus"] if len(self.code_history) >= 2 else 0.0
        conflict_penalty = -policy["submit_conflict_penalty"] if self._excludes1_conflict_seen else 0.0

        score = self._grade_bounds(
            base
            + search_history_bonus
            + search_hit_bonus
            + code_history_bonus
            + depth_bonus
            + conflict_penalty
        )
        return score, grader_name, {
            "base": base,
            "exact_match": exact_match,
            "family_match": family_match,
            "search_hit": 1.0 if submitted_code in self._last_search_codes else 0.0,
            "search_hit_bonus": search_hit_bonus,
            "search_history_bonus": search_history_bonus,
            "code_history_bonus": code_history_bonus,
            "depth_bonus": depth_bonus,
            "conflict_penalty": conflict_penalty,
            "final": score,
        }

    def _build_grading_trace(
        self,
        action_type: str,
        grader_used: str,
        reward: float,
        reward_components: dict[str, float],
    ) -> GradingTrace:
        task_id = self._current_task["task_id"] if self._current_task else ""
        difficulty = self._current_task["difficulty"] if self._current_task else ""
        return GradingTrace(
            action_type=action_type,
            grader=grader_used,
            difficulty=difficulty,
            step=self._state.step_count,
            task_id=task_id,
            reward=reward,
            reward_components={
                key: value
                for key, value in reward_components.items()
                if key not in self._SENSITIVE_REWARD_COMPONENT_KEYS
            },
            search_history=list(self.search_history),
            code_history=list(self.code_history),
            last_search_codes=sorted(self._last_search_codes),
            excludes1_conflict_seen=self._excludes1_conflict_seen,
        )

    def _v2_tasks_dir(self) -> Path:
        return Path(__file__).resolve().parent.parent / "examples"

    def _load_tasks(self) -> list[dict[str, Any]]:
        tasks_path = Path(__file__).resolve().parent.parent / "tasks.json"
        with tasks_path.open("r", encoding="utf-8") as fh:
            tasks = json.load(fh)
        return tasks

    def _load_v2_tasks(self) -> list[dict[str, Any]]:
        tasks: list[dict[str, Any]] = []
        v2_tasks_dir = self._v2_tasks_dir()
        if not v2_tasks_dir.exists():
            return tasks

        seen_task_ids: set[str] = set()
        for task_path in sorted(v2_tasks_dir.glob("v2_task*.json")):
            try:
                with task_path.open("r", encoding="utf-8") as fh:
                    raw_payload = json.load(fh)
            except (OSError, ValueError):
                continue

            candidate_tasks = raw_payload if isinstance(raw_payload, list) else [raw_payload]
            for payload in candidate_tasks:
                if not isinstance(payload, dict):
                    continue
                task_id = payload.get("task_id")
                if not isinstance(task_id, str) or not task_id or task_id in seen_task_ids:
                    continue
                seen_task_ids.add(task_id)
                tasks.append(payload)

        return tasks

    def _reset_ehr_state(self) -> None:
        self._ehr_modules = self._current_task.get("ehr_modules", {}) if self._current_task else {}
        self._ehr_remaining_budget = {
            module_name: max(int(module.get("query_budget", 0)), 0)
            for module_name, module in self._ehr_modules.items()
        }
        self._revealed_evidence = {}
        self._last_error = None
        self._invalid_reason = None

    def _reset_reasoning_state(self) -> None:
        self._reasoning_log = None
        self._reasoning_log_verified = False
        self._schema_validation_passed = False
        self._last_validated_schema_version = None

    def _ehr_map(self) -> dict[str, EhrModuleState]:
        ehr_map: dict[str, EhrModuleState] = {}
        for module_name, module in self._ehr_modules.items():
            remaining = self._ehr_remaining_budget.get(module_name, 0)
            revealed_count = sum(
                1 for snippet in self._revealed_evidence.values() if snippet.module == module_name
            )
            if remaining <= 0:
                status = "exhausted"
            elif revealed_count > 0:
                status = "open"
            else:
                status = module.get("status", "closed")
            ehr_map[module_name] = EhrModuleState(
                status=status,
                query_budget_remaining=remaining,
                revealed_count=revealed_count,
            )
        return ehr_map

    def _requires_reasoning_log(self) -> bool:
        if not self._current_task:
            return False
        claim_schema = (
            self._current_task.get("policy_rules", {}).get("claim_schema", {})
            if isinstance(self._current_task.get("policy_rules", {}), dict)
            else {}
        )
        required_fields = {
            str(field_name).strip()
            for field_name in claim_schema.get("required_fields", [])
            if str(field_name).strip()
        }
        return "reasoning_log_id" in required_fields or bool(self._current_task.get("target_evidence"))

    def _observation(
        self,
        *,
        reward: float,
        done: bool,
        grading: GradingTrace | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MedicalObservation:
        return MedicalObservation(
            patient_note=self._patient_note,
            search_results=self._search_results,
            detailed_info=self._detailed_info,
            current_reward=reward,
            done=done,
            reward=reward,
            grading=grading or GradingTrace(),
            ehr_map=self._ehr_map(),
            revealed_evidence=list(self._revealed_evidence.values()),
            policy_state=self._policy_engine.policy_state(),
            drift_notice=self._policy_engine.drift_notice,
            reasoning_log=self._reasoning_log,
            reasoning_log_verified=self._reasoning_log_verified,
            reward_metrics=self._reward_metrics or RewardMetrics(),
            last_error=self._last_error,
            invalid_reason=self._invalid_reason,
            metadata=metadata or {},
        )

    def _invalid_observation(
        self,
        *,
        action_type: str,
        error: str,
        reason: str,
        done: bool,
        metadata: dict[str, Any] | None = None,
    ) -> MedicalObservation:
        self._last_error = error
        self._invalid_reason = reason
        self._detailed_info = reason
        grading = self._build_grading_trace(
            action_type=action_type,
            grader_used=self._grader_name(),
            reward=0.0,
            reward_components={
                "base": 0.0,
                "invalid_action": 1.0,
                "final": 0.0,
            },
        )
        return self._observation(
            reward=0.0,
            done=done,
            grading=grading,
            metadata=metadata or {},
        )

    def _evidence_matches_query(self, evidence: dict[str, Any], query: str) -> bool:
        normalized_query = query.strip().lower()
        if not normalized_query:
            return True

        haystack_parts = [
            str(evidence.get("evidence_id", "")),
            str(evidence.get("text", "")),
            " ".join(str(code) for code in evidence.get("supports_codes", [])),
        ]
        haystack = " ".join(haystack_parts).lower()
        return any(term in haystack for term in normalized_query.split())

    def _query_ehr(self, module_name: str | None, query: str) -> tuple[float, dict[str, float]]:
        if not module_name:
            self._last_error = "missing_module"
            self._invalid_reason = "QUERY_EHR requires a module."
            self._detailed_info = self._invalid_reason
            return 0.0, {"base": 0.0, "invalid_action": 1.0, "final": 0.0}

        module = self._ehr_modules.get(module_name)
        if module is None:
            self._last_error = "invalid_module"
            self._invalid_reason = f"Unknown EHR module: {module_name}"
            self._detailed_info = self._invalid_reason
            return 0.0, {"base": 0.0, "invalid_action": 1.0, "final": 0.0}

        remaining = self._ehr_remaining_budget.get(module_name, 0)
        if remaining <= 0:
            self._last_error = "query_budget_exhausted"
            self._invalid_reason = f"EHR module query budget exhausted: {module_name}"
            self._detailed_info = self._invalid_reason
            return 0.0, {"base": 0.0, "budget_exhausted": 1.0, "final": 0.0}

        self._last_error = None
        self._invalid_reason = None
        self._ehr_remaining_budget[module_name] = remaining - 1

        matched: list[EvidenceSnippet] = []
        for raw_evidence in module.get("evidence", []):
            evidence_id = str(raw_evidence.get("evidence_id", ""))
            if not evidence_id or evidence_id in self._revealed_evidence:
                continue
            if not self._evidence_matches_query(raw_evidence, query):
                continue
            known_fields = {"evidence_id", "text", "supports_codes"}
            snippet = EvidenceSnippet(
                evidence_id=evidence_id,
                module=module_name,
                text=str(raw_evidence.get("text", "")),
                supports_codes=list(raw_evidence.get("supports_codes", [])),
                metadata={
                    key: value
                    for key, value in raw_evidence.items()
                    if key not in known_fields
                },
            )
            self._revealed_evidence[evidence_id] = snippet
            matched.append(snippet)

        self._detailed_info = f"QUERY_EHR revealed {len(matched)} evidence snippet(s) from {module_name}."
        reward = 0.03 if matched else 0.0
        return reward, {
            "base": 0.0,
            "evidence_revealed": float(len(matched)),
            "query_budget_remaining": float(self._ehr_remaining_budget[module_name]),
            "final": reward,
        }

    def verify_reasoning_log(
        self,
        payload: ReasoningLogPayload | None,
    ) -> tuple[bool, ReasoningLog | None, str | None, str | None]:
        if payload is None:
            return False, None, "missing_reasoning_payload", "REASONING_LOG requires a structured payload."

        candidate_code = payload.candidate_code.strip()
        if not candidate_code:
            return False, None, "missing_candidate_code", "REASONING_LOG requires a candidate_code."

        rationale = payload.rationale.strip()
        if not rationale:
            return False, None, "missing_rationale", "REASONING_LOG requires a rationale."

        evidence_ids = [item.strip() for item in payload.evidence_ids if item.strip()]
        if not evidence_ids:
            return False, None, "missing_evidence_ids", "REASONING_LOG requires at least one evidence id."

        unknown_evidence_ids = [
            evidence_id for evidence_id in evidence_ids if evidence_id not in self._revealed_evidence
        ]
        if unknown_evidence_ids:
            return (
                False,
                None,
                "unknown_evidence_ids",
                "REASONING_LOG cites evidence that has not been revealed: "
                + ", ".join(unknown_evidence_ids),
            )

        supporting_evidence_ids = [
            evidence_id
            for evidence_id in evidence_ids
            if candidate_code in self._revealed_evidence[evidence_id].supports_codes
        ]
        if not supporting_evidence_ids:
            return (
                False,
                None,
                "ungrounded_candidate_code",
                f"REASONING_LOG evidence does not support candidate code: {candidate_code}",
            )

        target_evidence = {
            str(item)
            for item in self._current_task.get("target_evidence", [])
            if str(item).strip()
        } if self._current_task else set()
        if target_evidence and candidate_code == self._target_code():
            cited_evidence = set(evidence_ids)
            missing_target_evidence = sorted(target_evidence - cited_evidence)
            if missing_target_evidence:
                return (
                    False,
                    None,
                    "missing_target_evidence",
                    "REASONING_LOG is missing required target evidence: "
                    + ", ".join(missing_target_evidence),
                )

        visible_policy_rules = {rule.rule_id for rule in self._policy_engine.policy_state().rules}
        unknown_policy_rule_ids = [
            rule_id for rule_id in payload.policy_rule_ids if rule_id not in visible_policy_rules
        ]
        if unknown_policy_rule_ids:
            return (
                False,
                None,
                "unknown_policy_rule_ids",
                "REASONING_LOG cites policy rules that are not visible: "
                + ", ".join(unknown_policy_rule_ids),
            )

        reasoning_log = ReasoningLog(
            reasoning_log_id=f"rl_{self._state.episode_id}_{self._state.step_count + 1}",
            candidate_code=candidate_code,
            rationale=rationale,
            evidence_ids=evidence_ids,
            policy_rule_ids=[item.strip() for item in payload.policy_rule_ids if item.strip()],
            verified=True,
        )
        return True, reasoning_log, None, None

    def _submit_reasoning_log(self, payload: ReasoningLogPayload | None) -> tuple[float, dict[str, float]]:
        verified, reasoning_log, last_error, invalid_reason = self.verify_reasoning_log(payload)

        fallback_payload = payload or ReasoningLogPayload()
        attempted_reasoning_log = reasoning_log or ReasoningLog(
            reasoning_log_id=f"rl_{self._state.episode_id}_{self._state.step_count}",
            candidate_code=fallback_payload.candidate_code.strip(),
            rationale=fallback_payload.rationale.strip(),
            evidence_ids=[item.strip() for item in fallback_payload.evidence_ids if item.strip()],
            policy_rule_ids=[item.strip() for item in fallback_payload.policy_rule_ids if item.strip()],
            verified=False,
        )

        self._reasoning_log = attempted_reasoning_log
        self._reasoning_log_verified = verified
        self._last_error = last_error
        self._invalid_reason = invalid_reason

        if not verified:
            self._detailed_info = invalid_reason or "REASONING_LOG verification failed."
            return 0.0, {
                "base": 0.0,
                "reasoning_verified": 0.0,
                "invalid_action": 1.0 if last_error == "missing_reasoning_payload" else 0.0,
                "final": 0.0,
            }

        self._detailed_info = (
            f"REASONING_LOG accepted {attempted_reasoning_log.reasoning_log_id} "
            f"for candidate code {attempted_reasoning_log.candidate_code}."
        )
        reward = 0.04
        return reward, {
            "base": 0.0,
            "reasoning_verified": 1.0,
            "evidence_count": float(len(attempted_reasoning_log.evidence_ids)),
            "final": reward,
        }

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> MedicalObservation:
        """
        Reset the environment.

        Args:
            seed: Optional seed for deterministic random task selection
            episode_id: Optional explicit episode identifier
            task_id: Optional explicit task identifier to load

        Returns:
            MedicalObservation with the initial patient note
        """
        task_id = kwargs.get("task_id")
        self._state = State(
            episode_id=episode_id if episode_id is not None else str(uuid4()),
            step_count=0,
        )
        selectable_tasks = self._all_tasks if self._all_tasks else self._tasks
        if task_id is None:
            task_selector = random if seed is None else random.Random(seed)
            self._current_task = task_selector.choice(selectable_tasks)
        else:
            self._current_task = next(
                (
                    task
                    for task in selectable_tasks
                    if task["task_id"] == task_id
                ),
                None,
            )
            if self._current_task is None:
                raise InvalidTaskIdError(task_id)
        self._patient_note = self._current_task.get("patient_note", "")
        self._search_results = []
        self._detailed_info = ""
        self.code_history = []
        self.search_history = []
        self._search_result_history = set()
        self._last_search_codes = set()
        self._excludes1_conflict_seen = False
        self._reset_ehr_state()
        self._reset_reasoning_state()
        self._policy_engine.reset(self._current_task)
        self._reward_engine.reset(self._current_task)
        self._reward_metrics = None
        self._episode_done = False

        return self._observation(reward=0.0, done=False)

    def step(
        self,
        action: MedicalAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> MedicalObservation:  # type: ignore[override]
        """
        Execute a workflow step.

        Args:
            action: MedicalAction describing the requested workflow operation
            timeout_s: Optional OpenEnv timeout hint (currently unused)

        Returns:
            MedicalObservation with search results, details, or submission status
        """
        if self._episode_done:
            return self._invalid_observation(
                action_type=action.action_type.value,
                error="terminal_misuse",
                reason="Episode already ended. Call reset() before taking another step.",
                done=True,
                metadata={
                    "query": action.query,
                    "module": action.module,
                    "step": self._state.step_count,
                    "timed_out": False,
                },
            )

        self._state.step_count += 1
        previous_schema_version = self._policy_engine.active_schema_version
        self._policy_engine.maybe_trigger_drift(self._state.step_count)
        if previous_schema_version != self._policy_engine.active_schema_version:
            self._schema_validation_passed = False

        reward = 0.0
        done = False
        excludes1_penalty = False
        grader_used = self._grader_name()
        reward_components: dict[str, float] = {"final": 0.0}
        self._last_error = None
        self._invalid_reason = None

        if action.action_type == MedicalActionType.SEARCH:
            if not action.query.strip():
                return self._invalid_observation(
                    action_type=action.action_type.value,
                    error="invalid_action",
                    reason="SEARCH requires a non-empty query.",
                    done=False,
                    metadata={
                        "query": action.query,
                        "module": action.module,
                        "step": self._state.step_count,
                        "timed_out": False,
                    },
                )
            self.search_history.append(action.query)
            self._search_results = [SearchResult(**result) for result in search_codes(action.query)]
            self._last_search_codes = {result.code for result in self._search_results}
            target_code = self._target_code()
            search_reward, reward_components = self._grade_search_step(
                query=action.query,
                result_codes=sorted(self._last_search_codes),
                target_code=target_code,
            )
            reward = search_reward
            if self._last_search_codes:
                self._search_result_history.add(tuple(sorted(self._last_search_codes)))
            self._detailed_info = f"Search returned {len(self._search_results)} candidate(s)"
        elif action.action_type == MedicalActionType.DETAILS:
            if not action.query.strip():
                return self._invalid_observation(
                    action_type=action.action_type.value,
                    error="invalid_action",
                    reason="DETAILS requires a non-empty code query.",
                    done=False,
                    metadata={
                        "query": action.query,
                        "module": action.module,
                        "step": self._state.step_count,
                        "timed_out": False,
                    },
                )
            details = get_code_details(action.query)
            if details:
                excludes = details.get("excludes", "")
                if any(previous_code in excludes for previous_code in self.code_history):
                    excludes1_penalty = True
                    self._excludes1_conflict_seen = True
                target_code = self._target_code()
                reward, reward_components = self._grade_details_step(
                    query=action.query,
                    details=details,
                    target_code=target_code,
                    conflict_seen=excludes1_penalty,
                )
                self.code_history.append(action.query)
                self._detailed_info = (
                    f"{details['long_desc']}\nExcludes: {excludes}"
                )
            else:
                self._last_error = "unknown_code"
                self._invalid_reason = f"Unknown code: {action.query}"
                self._detailed_info = self._invalid_reason
                reward_components = {
                    "base": 0.0,
                    "unknown_code": 1.0,
                    "final": 0.0,
                }
        elif action.action_type == MedicalActionType.SUBMIT:
            if not action.query.strip():
                return self._invalid_observation(
                    action_type=action.action_type.value,
                    error="invalid_action",
                    reason="SUBMIT requires a non-empty code query.",
                    done=False,
                    metadata={
                        "query": action.query,
                        "module": action.module,
                        "step": self._state.step_count,
                        "timed_out": False,
                    },
                )
            if not get_code_details(action.query):
                return self._invalid_observation(
                    action_type=action.action_type.value,
                    error="unknown_code",
                    reason=f"Unknown code: {action.query}",
                    done=False,
                    metadata={
                        "query": action.query,
                        "module": action.module,
                        "step": self._state.step_count,
                        "timed_out": False,
                    },
                )
            if self._requires_reasoning_log():
                if self._reasoning_log is None or not self._reasoning_log_verified:
                    self._last_error = "missing_reasoning_log"
                    self._invalid_reason = (
                        "SUBMIT requires a verified REASONING_LOG before final submission."
                    )
                    self._detailed_info = self._invalid_reason
                    reward = 0.0
                    reward_components = {
                        "base": 0.0,
                        "missing_reasoning_log": 1.0,
                        "final": 0.0,
                    }
                elif self._reasoning_log.candidate_code != action.query:
                    self._last_error = "reasoning_log_candidate_mismatch"
                    self._invalid_reason = (
                        "Verified REASONING_LOG candidate does not match submitted code: "
                        f"{self._reasoning_log.candidate_code} != {action.query}"
                    )
                    self._detailed_info = self._invalid_reason
                    reward = 0.0
                    reward_components = {
                        "base": 0.0,
                        "reasoning_log_candidate_mismatch": 1.0,
                        "final": 0.0,
                    }
                else:
                    target_code = self._target_code()
                    reward, grader_used, reward_components = self._grade_submission(action.query, target_code)
                    self._detailed_info = f"Submitted coding decision for query: {action.query}"
                    done = True
            else:
                target_code = self._target_code()
                reward, grader_used, reward_components = self._grade_submission(action.query, target_code)
                self._detailed_info = f"Submitted coding decision for query: {action.query}"
                done = True
        elif action.action_type == MedicalActionType.QUERY_EHR:
            reward, reward_components = self._query_ehr(action.module, action.query)
        elif action.action_type == MedicalActionType.CHECK_POLICY:
            result = self._policy_engine.check_policy()
            reward = result.reward
            reward_components = result.reward_components
            self._detailed_info = result.detailed_info
            self._last_error = result.last_error
            self._invalid_reason = result.invalid_reason
        elif action.action_type == MedicalActionType.VALIDATE_CLAIM_SCHEMA:
            result = self._policy_engine.validate_claim(action.payload, self._revealed_evidence)
            reward = result.reward
            reward_components = result.reward_components
            self._detailed_info = result.detailed_info
            self._last_error = result.last_error
            self._invalid_reason = result.invalid_reason
            self._schema_validation_passed = result.last_error is None and reward > 0.0
            if action.payload is not None:
                self._last_validated_schema_version = action.payload.schema_version.strip() or None
            else:
                self._last_validated_schema_version = None
        elif action.action_type == MedicalActionType.REASONING_LOG:
            reward, reward_components = self._submit_reasoning_log(action.payload)
        else:
            return self._invalid_observation(
                action_type=str(action.action_type),
                error="invalid_action",
                reason=f"Unsupported action type: {action.action_type}",
                done=False,
                metadata={
                    "query": action.query,
                    "module": action.module,
                    "step": self._state.step_count,
                    "timed_out": False,
                },
            )

        timed_out = self._state.step_count >= self._MAX_EPISODE_STEPS and action.action_type != MedicalActionType.SUBMIT
        if timed_out:
            reward = 0.0
            self._last_error = "timeout"
            self._invalid_reason = "Episode ended without a final SUBMIT decision."
            reward_components = {
                **reward_components,
                "timeout_penalty": 1.0,
                "final": reward,
            }
            done = True
            self._detailed_info = self._invalid_reason
        elif self._state.step_count >= self._MAX_EPISODE_STEPS:
            done = True

        self._episode_done = done
        evaluation = self._reward_engine.evaluate(
            action_type=action.action_type.value,
            base_reward=reward,
            reward_components=reward_components,
            step_count=self._state.step_count,
            search_history=self.search_history,
            code_history=self.code_history,
            last_error=self._last_error,
            invalid_reason=self._invalid_reason,
            reasoning_log=self._reasoning_log,
            reasoning_log_verified=self._reasoning_log_verified,
            schema_validation_passed=self._schema_validation_passed,
            validated_schema_version=self._last_validated_schema_version,
            drift_notice=self._policy_engine.drift_notice,
            policy_engine=self._policy_engine,
            timed_out=timed_out,
        )
        reward = evaluation.reward
        self._reward_metrics = evaluation.reward_metrics
        if action.action_type == MedicalActionType.SUBMIT and self._last_error in {
            "missing_reasoning_log",
            "reasoning_log_candidate_mismatch",
        }:
            reward = 0.0
            self._reward_metrics.final = 0.0

        grading = self._build_grading_trace(
            action_type=action.action_type.value,
            grader_used=grader_used,
            reward=reward,
            reward_components=reward_components,
        )

        return self._observation(
            reward=reward,
            done=done,
            grading=grading,
            metadata={
                "query": action.query,
                "module": action.module,
                "step": self._state.step_count,
                "excludes1_penalty": excludes1_penalty,
                "timed_out": timed_out,
            },
        )

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
