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
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import MedicalAction, MedicalActionType, MedicalObservation, SearchResult
    from .engine import get_code_details, initialize_db, search_codes
except ImportError:
    from models import MedicalAction, MedicalActionType, MedicalObservation, SearchResult
    from server.engine import get_code_details, initialize_db, search_codes


class RvedaEnvironment(Environment):
    """
    A task-driven medical-coding environment.
    """

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    _MIN_OPEN_SCORE = 0.05
    _MAX_OPEN_SCORE = 0.95

    def __init__(self):
        """Initialize the Rveda environment."""
        initialize_db()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._tasks = self._load_tasks()
        self._current_task: dict[str, str] | None = None
        self._patient_note = ""
        self._search_results: list[SearchResult] = []
        self._detailed_info = ""
        self.code_history: list[str] = []
        self.search_history: list[str] = []
        self._last_search_codes: set[str] = set()
        self._excludes1_conflict_seen = False

    def _grade_bounds(self, score: float) -> float:
        """Clamp score into an open interval so it is never exactly 0 or 1."""
        return min(max(score, self._MIN_OPEN_SCORE), self._MAX_OPEN_SCORE)

    def _same_family(self, submitted_code: str, target_code: str) -> bool:
        return bool(submitted_code and target_code and submitted_code[:3] == target_code[:3])

    def _grade_easy(self, submitted_code: str, target_code: str) -> float:
        if submitted_code == target_code:
            return 0.90
        if self._same_family(submitted_code, target_code):
            return 0.60
        if submitted_code in self._last_search_codes:
            return 0.35
        return 0.10

    def _grade_medium(self, submitted_code: str, target_code: str) -> float:
        if submitted_code == target_code:
            base = 0.72
        elif self._same_family(submitted_code, target_code):
            base = 0.42
        else:
            base = 0.12

        evidence_bonus = 0.0
        if self.search_history:
            evidence_bonus += 0.08
        if submitted_code in self._last_search_codes:
            evidence_bonus += 0.06
        if submitted_code in self.code_history:
            evidence_bonus += 0.07
        if self._excludes1_conflict_seen:
            evidence_bonus -= 0.10

        return self._grade_bounds(base + evidence_bonus)

    def _grade_hard(self, submitted_code: str, target_code: str) -> float:
        if submitted_code == target_code:
            base = 0.62
        elif self._same_family(submitted_code, target_code):
            base = 0.30
        else:
            base = 0.08

        process_bonus = 0.0
        if self.search_history:
            process_bonus += 0.08
        if submitted_code in self._last_search_codes:
            process_bonus += 0.08
        if submitted_code in self.code_history:
            process_bonus += 0.10
        if len(self.code_history) >= 2:
            process_bonus += 0.05
        if self._excludes1_conflict_seen:
            process_bonus -= 0.18

        return self._grade_bounds(base + process_bonus)

    def _grade_submission(self, submitted_code: str, target_code: str) -> tuple[float, str]:
        difficulty = (
            (self._current_task.get("difficulty", "medium").lower() if self._current_task else "medium")
        )

        if not target_code:
            return self._grade_bounds(0.10), difficulty

        if difficulty == "easy":
            score = self._grade_easy(submitted_code, target_code)
        elif difficulty == "hard":
            score = self._grade_hard(submitted_code, target_code)
        else:
            score = self._grade_medium(submitted_code, target_code)

        return self._grade_bounds(score), difficulty

    def _load_tasks(self) -> list[dict[str, str]]:
        tasks_path = Path(__file__).resolve().parent.parent / "tasks.json"
        with tasks_path.open("r", encoding="utf-8") as fh:
            tasks = json.load(fh)
        return tasks

    def reset(self, task_id: str | None = None) -> MedicalObservation:
        """
        Reset the environment.

        Args:
            task_id: Optional explicit task identifier to load

        Returns:
            MedicalObservation with the initial patient note
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        if task_id is None:
            self._current_task = random.choice(self._tasks)
        else:
            self._current_task = next(
                task for task in self._tasks if task["task_id"] == task_id
            )
        self._patient_note = self._current_task["patient_note"]
        self._search_results = []
        self._detailed_info = ""
        self.code_history = []
        self.search_history = []
        self._last_search_codes = set()
        self._excludes1_conflict_seen = False

        return MedicalObservation(
            patient_note=self._patient_note,
            search_results=self._search_results,
            detailed_info=self._detailed_info,
            current_reward=0.0,
            done=False,
            reward=0.0,
        )

    def step(self, action: MedicalAction) -> MedicalObservation:  # type: ignore[override]
        """
        Execute a workflow step.

        Args:
            action: MedicalAction describing the requested workflow operation

        Returns:
            MedicalObservation with search results, details, or submission status
        """
        self._state.step_count += 1

        reward = 0.0
        done = False
        excludes1_penalty = False
        grader_used = None

        if action.action_type == MedicalActionType.SEARCH:
            self.search_history.append(action.query)
            self._search_results = [SearchResult(**result) for result in search_codes(action.query)]
            self._last_search_codes = {result.code for result in self._search_results}
            self._detailed_info = ""
        elif action.action_type == MedicalActionType.DETAILS:
            details = get_code_details(action.query)
            if details:
                excludes = details.get("excludes", "")
                if any(previous_code in excludes for previous_code in self.code_history):
                    excludes1_penalty = True
                    self._excludes1_conflict_seen = True
                self.code_history.append(action.query)
                self._detailed_info = (
                    f"{details['long_desc']}\nExcludes: {excludes}"
                )
            else:
                self._detailed_info = ""
        elif action.action_type == MedicalActionType.SUBMIT:
            target_code = self._current_task["target_code"] if self._current_task else ""
            reward, grader_used = self._grade_submission(action.query, target_code)
            self._detailed_info = f"Submitted coding decision for query: {action.query}"
            done = True

        if self._state.step_count >= 10:
            done = True

        return MedicalObservation(
            patient_note=self._patient_note,
            search_results=self._search_results,
            detailed_info=self._detailed_info,
            current_reward=reward,
            done=done,
            reward=reward,
            metadata={
                "query": action.query,
                "step": self._state.step_count,
                "task_id": self._current_task["task_id"] if self._current_task else None,
                "difficulty": self._current_task["difficulty"] if self._current_task else None,
                "code_history": list(self.code_history),
                "search_history": list(self.search_history),
                "last_search_codes": sorted(self._last_search_codes),
                "excludes1_penalty": excludes1_penalty,
                "excludes1_conflict_seen": self._excludes1_conflict_seen,
                "grader_used": grader_used,
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
