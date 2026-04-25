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
        SearchResult,
    )
    from .engine import get_code_details, initialize_db, search_codes
except ImportError:
    from models import (
        EhrModuleState,
        EvidenceSnippet,
        GradingTrace,
        MedicalAction,
        MedicalActionType,
        MedicalObservation,
        SearchResult,
    )
    from server.engine import get_code_details, initialize_db, search_codes


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
        self._v2_example_tasks = self._load_v2_example_tasks()
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
            reward_components=reward_components,
            search_history=list(self.search_history),
            code_history=list(self.code_history),
            last_search_codes=sorted(self._last_search_codes),
            excludes1_conflict_seen=self._excludes1_conflict_seen,
        )

    def _v2_example_tasks_path(self) -> Path:
        return Path(__file__).resolve().parent.parent / "examples" / "v2_task_minimal.json"

    def _load_tasks(self) -> list[dict[str, Any]]:
        tasks_path = Path(__file__).resolve().parent.parent / "tasks.json"
        with tasks_path.open("r", encoding="utf-8") as fh:
            tasks = json.load(fh)
        return tasks

    def _load_v2_example_tasks(self) -> list[dict[str, Any]]:
        v2_example_path = self._v2_example_tasks_path()
        if v2_example_path.exists():
            with v2_example_path.open("r", encoding="utf-8") as fh:
                return [json.load(fh)]
        return []

    def _reset_ehr_state(self) -> None:
        self._ehr_modules = self._current_task.get("ehr_modules", {}) if self._current_task else {}
        self._ehr_remaining_budget = {
            module_name: max(int(module.get("query_budget", 0)), 0)
            for module_name, module in self._ehr_modules.items()
        }
        self._revealed_evidence = {}
        self._last_error = None
        self._invalid_reason = None

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
            last_error=self._last_error,
            invalid_reason=self._invalid_reason,
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
        if task_id is None:
            task_selector = random if seed is None else random.Random(seed)
            self._current_task = task_selector.choice(self._tasks)
        else:
            self._current_task = next(
                (
                    task
                    for task in [*self._tasks, *self._v2_example_tasks]
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
        self._state.step_count += 1

        reward = 0.0
        done = False
        excludes1_penalty = False
        grader_used = self._grader_name()
        reward_components: dict[str, float] = {"final": 0.0}
        self._last_error = None
        self._invalid_reason = None

        if action.action_type == MedicalActionType.SEARCH:
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
                self._detailed_info = ""
        elif action.action_type == MedicalActionType.SUBMIT:
            target_code = self._target_code()
            reward, grader_used, reward_components = self._grade_submission(action.query, target_code)
            self._detailed_info = f"Submitted coding decision for query: {action.query}"
            done = True
        elif action.action_type == MedicalActionType.QUERY_EHR:
            reward, reward_components = self._query_ehr(action.module, action.query)

        timed_out = self._state.step_count >= self._MAX_EPISODE_STEPS and action.action_type != MedicalActionType.SUBMIT
        if timed_out:
            reward = 0.0
            reward_components = {
                **reward_components,
                "timeout_penalty": 1.0,
                "final": reward,
            }
            done = True
            self._detailed_info = "Episode ended without a final SUBMIT decision."
        elif self._state.step_count >= self._MAX_EPISODE_STEPS:
            done = True

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
                "task_id": self._current_task["task_id"] if self._current_task else None,
                "difficulty": self._current_task["difficulty"] if self._current_task else None,
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
