"""Minimal training bridge for live Rveda rollouts.

This adapter keeps the environment interaction surface small enough for a first
TRL/Unsloth smoke run:

- `reset()` returns the current observation plus a prompt string
- `step()` advances the live environment with a structured action
- `rollout()` runs a tiny policy loop over the live environment

The bridge can wrap either:
- the in-process `RvedaEnvironment`, or
- an OpenEnv synchronous client such as `RvedaEnv(...).sync()`
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from typing import Any, Callable

from openenv.core.env_server.types import State

try:
    from .models import MedicalAction, MedicalObservation
    from .server.rveda_environment import RvedaEnvironment
except ImportError:
    from models import MedicalAction, MedicalObservation
    from server.rveda_environment import RvedaEnvironment


ActionInput = MedicalAction | dict[str, Any]
PolicyFn = Callable[[str, MedicalObservation], ActionInput]


@dataclass
class BridgeStep:
    """Single trainer-facing observation emitted by the bridge."""

    prompt: str
    observation: MedicalObservation
    reward: float
    done: bool
    info: dict[str, Any]
    action: dict[str, Any] | None = None


@dataclass
class RolloutTrace:
    """Compact rollout record for smoke training runs."""

    steps: list[BridgeStep]
    total_reward: float
    done: bool


class RvedaTrainingBridge:
    """Tiny adapter that exposes the live environment as trainer-friendly steps."""

    def __init__(self, env: Any | None = None):
        self._env = env if env is not None else RvedaEnvironment()
        self._connected = False

    def __enter__(self) -> "RvedaTrainingBridge":
        self._ensure_connected()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _ensure_connected(self) -> None:
        connector = getattr(self._env, "connect", None)
        if callable(connector) and not self._connected:
            connector()
            self._connected = True

    def close(self) -> None:
        closer = getattr(self._env, "close", None)
        if callable(closer):
            closer()
        self._connected = False

    def reset(
        self,
        *,
        task_id: str | None = None,
        seed: int | None = None,
        episode_id: str | None = None,
    ) -> BridgeStep:
        """Reset the live environment and return the first trainer-facing step."""
        self._ensure_connected()
        kwargs: dict[str, Any] = {}
        if task_id is not None:
            kwargs["task_id"] = task_id
        if seed is not None:
            kwargs["seed"] = seed
        if episode_id is not None:
            kwargs["episode_id"] = episode_id
        result = self._env.reset(**kwargs)
        return self._bridge_step(result, action=None)

    def step(self, action: ActionInput) -> BridgeStep:
        """Step the live environment with a structured action."""
        self._ensure_connected()
        medical_action = self._coerce_action(action)
        result = self._env.step(medical_action)
        return self._bridge_step(result, action=self._action_dict(medical_action))

    def step_json(self, action_json: str) -> BridgeStep:
        """Parse a JSON action and step the live environment."""
        payload = json.loads(action_json)
        if not isinstance(payload, dict):
            raise ValueError("Action JSON must decode to an object.")
        return self.step(payload)

    def rollout(
        self,
        policy: PolicyFn,
        *,
        task_id: str | None = None,
        seed: int | None = None,
        episode_id: str | None = None,
        max_steps: int | None = None,
    ) -> RolloutTrace:
        """Run a minimal live rollout suitable for a first training smoke test."""
        first_step = self.reset(task_id=task_id, seed=seed, episode_id=episode_id)
        steps = [first_step]
        total_reward = 0.0
        current = first_step

        while not current.done and (max_steps is None or len(steps) - 1 < max_steps):
            next_action = policy(current.prompt, current.observation)
            current = self.step(next_action)
            total_reward += current.reward
            steps.append(current)

        return RolloutTrace(
            steps=steps,
            total_reward=total_reward,
            done=steps[-1].done,
        )

    @staticmethod
    def format_observation(observation: MedicalObservation) -> str:
        """Render a compact text prompt from the current observation."""
        search_results = [result.model_dump() for result in observation.search_results]
        revealed_evidence = [snippet.model_dump() for snippet in observation.revealed_evidence]
        ehr_map = {
            module_name: module_state.model_dump()
            for module_name, module_state in observation.ehr_map.items()
        }

        return textwrap.dedent(
            f"""
            Patient Note: {observation.patient_note}
            EHR Map: {json.dumps(ehr_map, ensure_ascii=True, sort_keys=True)}
            Revealed Evidence: {json.dumps(revealed_evidence, ensure_ascii=True, sort_keys=True)}
            Search Results: {json.dumps(search_results, ensure_ascii=True, sort_keys=True)}
            Detailed Info: {observation.detailed_info}
            Last Error: {observation.last_error}
            Invalid Reason: {observation.invalid_reason}
            """
        ).strip()

    def _bridge_step(self, result: Any, *, action: dict[str, Any] | None) -> BridgeStep:
        observation, reward, done = self._unwrap_result(result)
        state = self._state()
        info = {
            "episode_id": getattr(state, "episode_id", None),
            "step_count": getattr(state, "step_count", None),
            "task_id": observation.grading.task_id,
            "last_error": observation.last_error,
            "invalid_reason": observation.invalid_reason,
            "metadata": dict(observation.metadata),
            "v2_verifier_snapshot": self.compute_v2_verifier_metrics(observation),
        }
        return BridgeStep(
            prompt=self.format_observation(observation),
            observation=observation,
            reward=reward,
            done=done,
            info=info,
            action=action,
        )

    def _state(self) -> State | Any:
        state_attr = getattr(self._env, "state", None)
        if callable(state_attr):
            return state_attr()
        return state_attr

    @staticmethod
    def _unwrap_result(result: Any) -> tuple[MedicalObservation, float, bool]:
        if hasattr(result, "observation"):
            observation = RvedaTrainingBridge._coerce_observation(result.observation)
            reward = float(result.reward or 0.0)
            done = bool(result.done)
            return observation, reward, done

        if isinstance(result, dict):
            if "observation" in result:
                observation = RvedaTrainingBridge._coerce_observation(result["observation"])
                reward = float(result.get("reward", 0.0) or 0.0)
                done = bool(result.get("done", False))
                return observation, reward, done
            observation = RvedaTrainingBridge._coerce_observation(result)
            reward = float(result.get("reward", 0.0) or 0.0)
            done = bool(result.get("done", False))
            return observation, reward, done

        if isinstance(result, (tuple, list)):
            if not result:
                raise ValueError("Empty step result is not supported.")
            observation = RvedaTrainingBridge._coerce_observation(result[0])
            reward = 0.0
            done = False
            if len(result) >= 5:
                reward = float(result[1] or 0.0)
                done = bool(result[2]) or bool(result[3])
            elif len(result) >= 4:
                reward = float(result[1] or 0.0)
                done = bool(result[2])
            elif len(result) >= 2 and isinstance(result[1], dict):
                reward = float(result[1].get("reward", 0.0) or 0.0)
                done = bool(result[1].get("done", False))
            return observation, reward, done

        reward = float(getattr(result, "reward", 0.0) or 0.0)
        done = bool(getattr(result, "done", False))
        return RvedaTrainingBridge._coerce_observation(result), reward, done

    @staticmethod
    def _coerce_observation(value: Any) -> MedicalObservation:
        if isinstance(value, MedicalObservation):
            return value
        if isinstance(value, dict):
            return MedicalObservation.model_validate(value)
        if hasattr(value, "model_dump"):
            return MedicalObservation.model_validate(value.model_dump())
        return MedicalObservation.model_validate(value)

    @staticmethod
    def compute_v2_verifier_metrics(
        observation: MedicalObservation,
        *,
        action: dict[str, Any] | None = None,
        fallback_reward: float = 0.0,
    ) -> dict[str, float]:
        """Return a small live-payload verifier snapshot for future RLVR wiring."""
        evidence_count = float(len(observation.revealed_evidence))
        search_result_count = float(len(observation.search_results))
        module_count = float(len(observation.ehr_map))
        invalid_flag = 1.0 if observation.last_error or observation.invalid_reason else 0.0

        action_type = str((action or {}).get("action_type", "")).strip().upper()
        module_name = str((action or {}).get("module", "")).strip()
        query = str((action or {}).get("query", "")).strip().lower()

        matched_evidence = 0.0
        if query:
            for snippet in observation.revealed_evidence:
                haystack = f"{snippet.evidence_id} {snippet.text} {' '.join(snippet.supports_codes)}".lower()
                if query in haystack:
                    matched_evidence = 1.0
                    break

        ehr_visible = 1.0 if observation.ehr_map else 0.0
        module_valid = 1.0 if not module_name or module_name in observation.ehr_map else 0.0
        grounding_proxy = min(1.0, evidence_count + matched_evidence)
        vetted_before_submit = 1.0 if action_type != "SUBMIT" or evidence_count > 0.0 else 0.0
        evidence_to_submission_ratio = evidence_count / max(1.0, 1.0 if action_type == "SUBMIT" else 0.0 + evidence_count)

        training_reward = (
            float(fallback_reward)
            + 0.05 * grounding_proxy
            + 0.02 * search_result_count
            + 0.02 * ehr_visible
            - 0.05 * invalid_flag
        )
        if action_type == "SUBMIT" and evidence_count <= 0.0:
            training_reward -= 0.05

        return {
            "fallback_reward": float(fallback_reward),
            "training_reward": float(training_reward),
            "grounding_proxy": float(grounding_proxy),
            "evidence_count": evidence_count,
            "search_result_count": search_result_count,
            "module_count": module_count,
            "invalid_flag": invalid_flag,
            "vetted_before_submit_rate": float(vetted_before_submit),
            "evidence_to_submission_ratio": float(evidence_to_submission_ratio),
            "module_valid": float(module_valid),
        }

    @staticmethod
    def _action_dict(action: MedicalAction) -> dict[str, Any]:
        return {
            "action_type": action.action_type.value,
            "query": action.query,
            "module": action.module,
            "payload": action.payload.model_dump() if action.payload is not None else None,
        }

    @staticmethod
    def _coerce_action(action: ActionInput) -> MedicalAction:
        if isinstance(action, MedicalAction):
            return action
        if not isinstance(action, dict):
            raise TypeError("Action must be a MedicalAction or a dict.")
        return MedicalAction(
            action_type=str(action.get("action_type", "")).strip().upper(),
            query=str(action.get("query", "")),
            module=(
                str(action["module"]).strip()
                if action.get("module") is not None
                else None
            ),
            payload=action.get("payload"),
        )
