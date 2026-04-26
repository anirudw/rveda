"""Minimal live GRPO smoke runner for Rveda.

This script keeps the first training proof intentionally small:

- baseline rollout uses the live environment through `RvedaTrainingBridge`
- optional GRPO training uses live environment rewards, not a static label set
- outputs JSON artifacts for baseline and post-train comparison

Local environment-only smoke:
    python train_grpo_smoke.py --skip-train

Colab / GPU smoke training:
    pip install torch datasets accelerate trl unsloth
    python train_grpo_smoke.py --model-name Qwen/Qwen2.5-7B-Instruct
    python train_grpo_smoke.py --smoke-model qwen2.5-14b --task-ids v2_easy_overweight_schema_v1 --samples-per-task 1 --episodes 1 --train-steps 1 --max-episode-steps 1 --max-new-tokens 32
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .trl_bridge import BridgeStep, RolloutTrace, RvedaTrainingBridge
except ImportError:
    from trl_bridge import BridgeStep, RolloutTrace, RvedaTrainingBridge


QWEN25_7B_MODEL = "Qwen/Qwen2.5-7B-Instruct"
QWEN25_14B_MODEL = "Qwen/Qwen2.5-14B-Instruct"
DEFAULT_OUTPUT_DIR = "artifacts/grpo_smoke"
QWEN25_14B_OUTPUT_DIR = "artifacts/grpo_smoke_qwen25_14b"
DEFAULT_TASK_IDS = ["easy_endo_1"]
ACTION_TYPES = {
    "SEARCH",
    "DETAILS",
    "SUBMIT",
    "QUERY_EHR",
    "CHECK_POLICY",
    "VALIDATE_CLAIM_SCHEMA",
    "REASONING_LOG",
}
ACTION_INSTRUCTION = (
    "Return exactly one JSON object with keys action_type, query, optional "
    "module, and optional payload. Valid action_type values are SEARCH, DETAILS, "
    "SUBMIT, QUERY_EHR, CHECK_POLICY, VALIDATE_CLAIM_SCHEMA, REASONING_LOG. "
    "Do not add markdown."
)
SAFE_SEARCH_BY_TASK = {
    "easy_endo_1": "weight",
    "medium_endo_1": "thyroid",
    "hard_cardio_1": "heart attack",
    "v2_easy_overweight_schema_v1": "overweight",
}
SAFE_EHR_QUERY_BY_TASK = {
    "v2_easy_overweight_schema_v1": "BMI weight",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal live GRPO smoke runner for Rveda.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for JSON outputs.")
    parser.add_argument("--model-name", default=QWEN25_7B_MODEL, help="Base model for smoke GRPO.")
    parser.add_argument(
        "--smoke-model",
        choices=["qwen2.5-7b", "qwen2.5-14b"],
        help="Named smoke preset. The 14B preset writes to a dedicated artifact folder unless --output-dir is set.",
    )
    parser.add_argument("--task-ids", nargs="+", default=DEFAULT_TASK_IDS, help="Task ids used for smoke train/eval.")
    parser.add_argument("--samples-per-task", type=int, default=4, help="Number of prompt rows per task.")
    parser.add_argument("--episodes", type=int, default=1, help="Evaluation episodes per task.")
    parser.add_argument("--max-episode-steps", type=int, default=3, help="Max steps during smoke evaluation.")
    parser.add_argument("--train-steps", type=int, default=2, help="Max GRPO optimization steps.")
    parser.add_argument("--max-new-tokens", type=int, default=96, help="Generation cap for action JSON.")
    parser.add_argument("--skip-train", action="store_true", help="Only run the live env baseline smoke.")
    parser.add_argument(
        "--disable-unsloth-fast-rl",
        action="store_true",
        help="Skip Unsloth's GRPO patch and use the plain TRL trainer path.",
    )
    parser.add_argument(
        "--emit-observability-only",
        action="store_true",
        help="Regenerate comparison JSON and plots from an existing completed output directory.",
    )
    args = parser.parse_args()
    if args.smoke_model == "qwen2.5-7b":
        args.model_name = QWEN25_7B_MODEL
    elif args.smoke_model == "qwen2.5-14b":
        args.model_name = QWEN25_14B_MODEL
        if args.output_dir == DEFAULT_OUTPUT_DIR:
            args.output_dir = QWEN25_14B_OUTPUT_DIR
    return args


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def build_command_metadata(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "argv": [sys.executable, *sys.argv],
        "model_name": args.model_name,
        "smoke_model": args.smoke_model,
        "output_dir": str(output_dir),
        "task_ids": args.task_ids,
        "samples_per_task": args.samples_per_task,
        "episodes": args.episodes,
        "max_episode_steps": args.max_episode_steps,
        "train_steps": args.train_steps,
        "max_new_tokens": args.max_new_tokens,
        "skip_train": args.skip_train,
        "disable_unsloth_fast_rl": args.disable_unsloth_fast_rl,
        "emit_observability_only": args.emit_observability_only,
        "expected_artifacts": [
            "command_metadata.json",
            "scripted_baseline.json",
            "train_rows_preview.json",
            "baseline_model_eval.json",
            "post_train_model_eval.json",
            "trainer_log_history.json",
            "summary.json",
            "baseline_vs_trained_comparison.json",
            "loss_plot.svg",
            "reward_plot.svg",
            "verifier_metrics_plot.svg",
        ],
    }


def save_failure_status(output_dir: Path, exc: Exception) -> None:
    save_json(
        output_dir / "run_status.json",
        {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "note": "This records a failed smoke attempt; it is not GRPO training proof.",
        },
    )


def save_failed_attempt_summary(
    output_dir: Path,
    args: argparse.Namespace,
    baseline: dict[str, Any],
    exc: Exception,
) -> None:
    save_json(
        output_dir / "summary.json",
        {
            "mode": "attempt_failed",
            "model_name": args.model_name,
            "smoke_model": args.smoke_model,
            "task_ids": args.task_ids,
            "disable_unsloth_fast_rl": args.disable_unsloth_fast_rl,
            "baseline_mean_total_reward": baseline["mean_total_reward"],
            "baseline_rollout_summary": baseline["rollout_summary"],
            "error_type": type(exc).__name__,
            "error": str(exc),
            "saved_files": [
                "command_metadata.json",
                "scripted_baseline.json",
                "run_status.json",
            ],
            "missing_artifacts": [
                "train_rows_preview.json",
                "baseline_model_eval.json",
                "post_train_model_eval.json",
            ],
        },
    )


def build_prompt(prompt: str) -> str:
    return f"{ACTION_INSTRUCTION}\n\n{prompt}"


def extract_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _latest_revealed_evidence_ids(observation) -> list[str]:
    return [snippet.evidence_id for snippet in observation.revealed_evidence]


def _target_code_from_runtime(runtime_path: Any) -> str:
    if not isinstance(runtime_path, dict):
        return ""
    return str(
        runtime_path.get("expected_submit_code")
        or runtime_path.get("expected_details_code")
        or ""
    ).strip()


def _schema_version_from_observation(observation) -> str:
    policy_state = getattr(observation, "policy_state", None)
    active_schema = getattr(policy_state, "active_schema_version", "")
    if active_schema:
        return str(active_schema)
    runtime_path = observation.metadata.get("current_runtime_path")
    if isinstance(runtime_path, dict):
        schema_version = runtime_path.get("expected_schema_version")
        if schema_version:
            return str(schema_version)
    return "v1"


def _claim_payload(observation, runtime_path: Any, *, include_reasoning_log: bool) -> dict[str, Any]:
    target_code = _target_code_from_runtime(runtime_path)
    evidence_ids = _latest_revealed_evidence_ids(observation)
    schema_version = _schema_version_from_observation(observation)
    payload: dict[str, Any] = {
        "diagnosis_code": target_code,
        "schema_version": schema_version,
        "evidence_ids": evidence_ids,
    }
    if include_reasoning_log:
        reasoning_log = getattr(observation, "reasoning_log", None)
        reasoning_log_id = getattr(reasoning_log, "reasoning_log_id", None)
        payload["reasoning_log_id"] = reasoning_log_id or "rl_pending"
    policy_state = getattr(observation, "policy_state", None)
    claim_schema = getattr(policy_state, "claim_schema", None)
    required_fields = set(getattr(claim_schema, "required_fields", []) or [])
    if "policy_attestations" in required_fields:
        payload["policy_attestations"] = ["policy_attestation_required"]
    return payload


def _candidate_code_from_observation(observation, runtime_path: Any) -> str:
    target_code = _target_code_from_runtime(runtime_path)
    if target_code:
        return target_code
    if observation.search_results:
        return str(observation.search_results[0].code).strip()
    return ""


def _visible_policy_rule_ids(observation) -> list[str]:
    policy_state = getattr(observation, "policy_state", None)
    visible_rules = getattr(policy_state, "rules", []) or []
    return [
        str(rule.rule_id).strip()
        for rule in visible_rules
        if getattr(rule, "rule_id", None) and str(rule.rule_id).strip()
    ]


def _first_query(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value:
        return str(value[0])
    return ""


def default_action_for_observation(observation, task_id: str) -> dict[str, Any]:
    runtime_path = observation.metadata.get("current_runtime_path")
    target_modules = (
        set(str(module) for module in runtime_path.get("recommended_ehr_modules", []))
        if isinstance(runtime_path, dict)
        else set()
    )
    if observation.ehr_map:
        for module_name, module_state in observation.ehr_map.items():
            if target_modules and module_name not in target_modules:
                continue
            if module_state.query_budget_remaining > 0 and module_state.revealed_count == 0:
                ehr_queries_by_module = (
                    runtime_path.get("recommended_ehr_queries_by_module", {})
                    if isinstance(runtime_path, dict)
                    else {}
                )
                ehr_queries = (
                    ehr_queries_by_module.get(module_name, [])
                    if isinstance(ehr_queries_by_module, dict)
                    else []
                )
                if not ehr_queries and isinstance(runtime_path, dict):
                    ehr_queries = runtime_path.get("recommended_ehr_queries", [])
                query = _first_query(ehr_queries)
                return {
                    "action_type": "QUERY_EHR",
                    "module": module_name,
                    "query": (
                        query
                        if query
                        else SAFE_EHR_QUERY_BY_TASK.get(task_id, "BMI weight")
                    ),
                }

    if observation.search_results:
        if observation.grading.action_type == "SEARCH":
            return {
                "action_type": "DETAILS",
                "query": observation.search_results[0].code,
            }
        target_code = _candidate_code_from_observation(observation, runtime_path)
        if target_code:
            policy_state = getattr(observation, "policy_state", None)
            policy_checked = bool(getattr(policy_state, "checked", False))
            if not policy_checked:
                return {"action_type": "CHECK_POLICY", "query": ""}
            validation_just_passed = (
                observation.grading.action_type == "VALIDATE_CLAIM_SCHEMA"
                and observation.last_error is None
                and observation.invalid_reason is None
            )
            validation_failed = (
                observation.grading.action_type == "VALIDATE_CLAIM_SCHEMA"
                and (observation.last_error is not None or observation.invalid_reason is not None)
            )
            if validation_failed or (
                getattr(observation, "reasoning_log", None) is None and not validation_just_passed
            ):
                return {
                    "action_type": "VALIDATE_CLAIM_SCHEMA",
                    "query": "",
                    "payload": _claim_payload(
                        observation,
                        runtime_path,
                        include_reasoning_log=True,
                    ),
                }
            if not getattr(observation, "reasoning_log_verified", False):
                return {
                    "action_type": "REASONING_LOG",
                    "query": "",
                    "payload": {
                        "candidate_code": target_code,
                        "rationale": "Synthetic smoke policy cites revealed evidence and ICD detail.",
                        "evidence_ids": _latest_revealed_evidence_ids(observation),
                        "policy_rule_ids": _visible_policy_rule_ids(observation),
                    },
                }
        return {
            "action_type": "SUBMIT",
            "query": target_code or observation.search_results[0].code,
        }

    search_queries = (
        runtime_path.get("recommended_search_queries", [])
        if isinstance(runtime_path, dict)
        else []
    )
    return {
        "action_type": "SEARCH",
        "query": (
            str(search_queries[0])
            if search_queries
            else SAFE_SEARCH_BY_TASK.get(task_id, "weight")
        ),
    }


def coerce_action_dict(raw_action: dict[str, Any], observation, task_id: str) -> tuple[dict[str, Any], bool]:
    action_type = str(raw_action.get("action_type", "")).strip().upper()
    query = str(raw_action.get("query", "")).strip()
    module = raw_action.get("module")
    payload = raw_action.get("payload")
    module_text = str(module).strip() if module is not None else None

    if action_type not in ACTION_TYPES:
        return default_action_for_observation(observation, task_id), True
    if action_type in {"SEARCH", "DETAILS", "SUBMIT", "QUERY_EHR"} and not query:
        return default_action_for_observation(observation, task_id), True
    if action_type == "QUERY_EHR" and not module_text:
        return default_action_for_observation(observation, task_id), True

    action = {
        "action_type": action_type,
        "query": query,
    }
    if module_text is not None:
        action["module"] = module_text
    if isinstance(payload, dict):
        action["payload"] = payload
    return action, False


def parse_model_action(text: str, observation, task_id: str) -> tuple[dict[str, Any], bool]:
    payload = extract_json_object(text)
    if payload is None:
        return default_action_for_observation(observation, task_id), True
    return coerce_action_dict(payload, observation, task_id)


def serialize_step(step: BridgeStep) -> dict[str, Any]:
    return {
        "action": step.action,
        "reward": step.reward,
        "done": step.done,
        "info": step.info,
        "observation": step.observation.model_dump(),
    }


def serialize_rollout(task_id: str, trace: RolloutTrace) -> dict[str, Any]:
    step_rewards = [step.reward for step in trace.steps if step.action is not None]
    nonzero_reward_steps = sum(1 for reward in step_rewards if reward != 0)
    submitted_steps = sum(
        1
        for step in trace.steps
        if step.action is not None and str(step.action.get("action_type", "")).upper() == "SUBMIT"
    )
    evidence_steps = sum(1 for step in trace.steps if len(step.observation.revealed_evidence) > 0)
    return {
        "task_id": task_id,
        "done": trace.done,
        "total_reward": trace.total_reward,
        "steps": [serialize_step(step) for step in trace.steps],
        "summary": {
            "episode_length": max(len(trace.steps) - 1, 0),
            "nonzero_reward_steps": nonzero_reward_steps,
            "submission_steps": submitted_steps,
            "evidence_steps": evidence_steps,
            "mean_step_reward": sum(step_rewards) / max(len(step_rewards), 1),
            "max_step_reward": max(step_rewards, default=0.0),
            "min_step_reward": min(step_rewards, default=0.0),
            "final_done": trace.done,
        },
    }


def evaluate_scripted_policy(task_ids: list[str], episodes: int, max_episode_steps: int) -> dict[str, Any]:
    results = []
    for task_id in task_ids:
        for _ in range(episodes):
            with RvedaTrainingBridge() as bridge:
                trace = bridge.rollout(
                    lambda prompt, obs: default_action_for_observation(obs, task_id),
                    task_id=task_id,
                    max_steps=max_episode_steps,
                )
            results.append(serialize_rollout(task_id, trace))
    mean_reward = sum(item["total_reward"] for item in results) / max(len(results), 1)
    nonzero_reward_rate = sum(1 for item in results if item["total_reward"] != 0) / max(len(results), 1)
    return {
        "policy": "scripted_baseline",
        "task_ids": task_ids,
        "episodes": results,
        "mean_total_reward": mean_reward,
        "rollout_summary": {
            "episode_count": len(results),
            "nonzero_reward_rate": nonzero_reward_rate,
            "mean_total_reward": mean_reward,
        },
    }


def build_train_rows(task_ids: list[str], samples_per_task: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with RvedaTrainingBridge() as bridge:
        for task_id in task_ids:
            for _ in range(samples_per_task):
                first_step = bridge.reset(task_id=task_id)
                rows.append(
                    {
                        "prompt": build_prompt(first_step.prompt),
                        "task_id": task_id,
                    }
                )
    return rows


def extract_completion_text(completion: Any) -> str:
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return str(completion[0].get("content", ""))
    return str(completion)


def make_live_reward_func():
    def reward_func(prompts, completions, task_id, **kwargs):
        rewards = []
        current_task_ids = _normalize_task_ids(task_id, len(completions))
        for completion, current_task_id in zip(completions, current_task_ids):
            with RvedaTrainingBridge() as bridge:
                first_step = bridge.reset(task_id=current_task_id)
                action, _used_fallback = parse_model_action(
                    extract_completion_text(completion),
                    first_step.observation,
                    current_task_id,
                )
                result = bridge.step(action)
                verifier_metrics = bridge.compute_v2_verifier_metrics(
                    result.observation,
                    action=action,
                    fallback_reward=float(result.reward),
                )
            rewards.append(float(verifier_metrics["training_reward"]))
        return rewards

    return reward_func


def _normalize_task_ids(task_ids: Any, count: int) -> list[str]:
    if count <= 0:
        return []
    if isinstance(task_ids, str):
        return [task_ids] * count
    if isinstance(task_ids, Sequence):
        normalized = [str(task_id) for task_id in task_ids]
        if not normalized:
            return [DEFAULT_TASK_IDS[0]] * count
        if len(normalized) >= count:
            return normalized[:count]
        return normalized + [normalized[-1]] * (count - len(normalized))
    return [str(task_ids)] * count


def _svg_escape(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _metric_or_none(value: Any) -> float | None:
    coerced = _safe_float(value)
    return coerced if coerced is not None else None


def _step_action_type(step: dict[str, Any]) -> str:
    action = step.get("action")
    if not isinstance(action, dict):
        return ""
    return str(action.get("action_type", "")).upper()


def _step_snapshot(step: dict[str, Any]) -> dict[str, Any]:
    info = step.get("info")
    if not isinstance(info, dict):
        return {}
    snapshot = info.get("v2_verifier_snapshot")
    return snapshot if isinstance(snapshot, dict) else {}


def _episode_timed_out(episode: dict[str, Any]) -> bool:
    for step in episode.get("steps", []):
        if not isinstance(step, dict):
            continue
        observation = step.get("observation")
        info = step.get("info")
        metadata = observation.get("metadata", {}) if isinstance(observation, dict) else {}
        info_metadata = info.get("metadata", {}) if isinstance(info, dict) else {}
        invalid_reason = observation.get("invalid_reason") if isinstance(observation, dict) else None
        last_error = observation.get("last_error") if isinstance(observation, dict) else None
        if metadata.get("timed_out") or info_metadata.get("timed_out"):
            return True
        if last_error == "timeout" or invalid_reason == "Episode ended without a final SUBMIT decision.":
            return True
    return False


def _mean_snapshot_metric(episodes: list[dict[str, Any]], metric_name: str) -> float | None:
    values: list[float] = []
    for episode in episodes:
        for step in episode.get("steps", []):
            if not isinstance(step, dict):
                continue
            value = _metric_or_none(_step_snapshot(step).get(metric_name))
            if value is not None:
                values.append(value)
    return _mean(values)


def policy_metrics(eval_payload: dict[str, Any]) -> dict[str, Any]:
    episodes = [episode for episode in eval_payload.get("episodes", []) if isinstance(episode, dict)]
    episode_count = len(episodes)
    total_rewards = [_metric_or_none(episode.get("total_reward")) for episode in episodes]
    total_rewards = [reward for reward in total_rewards if reward is not None]

    action_counts = {"SEARCH": 0, "DETAILS": 0, "SUBMIT": 0, "QUERY_EHR": 0}
    for episode in episodes:
        for step in episode.get("steps", []):
            if not isinstance(step, dict):
                continue
            action_type = _step_action_type(step)
            if action_type in action_counts:
                action_counts[action_type] += 1

    submit_count = action_counts["SUBMIT"]
    search_to_submission_ratio = (
        action_counts["SEARCH"] / submit_count
        if submit_count > 0
        else None
    )
    timeout_count = sum(1 for episode in episodes if _episode_timed_out(episode))
    grounding_proxy = _mean_snapshot_metric(episodes, "grounding_proxy")

    return {
        "episode_count": episode_count,
        "mean_total_reward": _mean(total_rewards),
        "total_rewards": total_rewards,
        "action_counts": action_counts,
        "search_to_submission_ratio": search_to_submission_ratio,
        "timeout_frequency": timeout_count / episode_count if episode_count else None,
        "grounding_f1": grounding_proxy,
        "grounding_f1_note": (
            "Proxy from v2_verifier_snapshot.grounding_proxy; true precision/recall labels are not emitted by this smoke runner."
        ),
        "drift_adaptation_rate": None,
        "drift_adaptation_rate_note": "Unavailable: current smoke task has drift disabled and emits no drift adaptation events.",
        "schema_validation_pass_rate": None,
        "schema_validation_pass_rate_note": "Unavailable: current smoke runner does not execute or emit explicit schema validation checks.",
        "verifier_metric_means": {
            key: _mean_snapshot_metric(episodes, key)
            for key in [
                "training_reward",
                "fallback_reward",
                "grounding_proxy",
                "evidence_count",
                "search_result_count",
                "module_count",
                "invalid_flag",
                "vetted_before_submit_rate",
                "evidence_to_submission_ratio",
                "module_valid",
            ]
        },
    }


def compare_metric(before: dict[str, Any], after: dict[str, Any], key: str) -> dict[str, Any]:
    before_value = before.get(key)
    after_value = after.get(key)
    delta = (
        float(after_value) - float(before_value)
        if isinstance(before_value, (int, float)) and isinstance(after_value, (int, float))
        else None
    )
    return {
        "baseline": before_value,
        "trained": after_value,
        "delta": delta,
    }


def build_comparison(
    baseline_eval: dict[str, Any],
    trained_eval: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    baseline_metrics = policy_metrics(baseline_eval)
    trained_metrics = policy_metrics(trained_eval)
    metric_keys = [
        "mean_total_reward",
        "grounding_f1",
        "drift_adaptation_rate",
        "search_to_submission_ratio",
        "timeout_frequency",
        "schema_validation_pass_rate",
    ]
    comparison = {
        key: compare_metric(baseline_metrics, trained_metrics, key)
        for key in metric_keys
    }
    comparison["verifier_metric_means"] = {
        key: compare_metric(
            baseline_metrics["verifier_metric_means"],
            trained_metrics["verifier_metric_means"],
            key,
        )
        for key in baseline_metrics["verifier_metric_means"]
    }
    return {
        "model_name": summary.get("model_name"),
        "smoke_model": summary.get("smoke_model"),
        "task_ids": summary.get("task_ids"),
        "train_steps": summary.get("train_steps"),
        "interpretation": (
            "Smoke observability artifact. Equal baseline/trained rewards indicate pipeline proof, not meaningful learning."
        ),
        "baseline_policy": baseline_metrics,
        "trained_policy": trained_metrics,
        "comparison": comparison,
        "trainer_metrics": summary.get("trainer_metrics", {}),
    }


def _plot_points(
    values: list[float],
    x0: float,
    y0: float,
    width: float,
    height: float,
    min_value: float,
    max_value: float,
) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return f"{x0 + width / 2:.2f},{y0 + height / 2:.2f}"
    span = max(max_value - min_value, 1e-9)
    points = []
    for idx, value in enumerate(values):
        x = x0 + (idx / (len(values) - 1)) * width
        y = y0 + height - ((value - min_value) / span) * height
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def write_svg_line_plot(
    path: Path,
    title: str,
    y_label: str,
    series: dict[str, list[float]],
    *,
    x_label: str = "Step",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 760
    height = 440
    plot_x = 80
    plot_y = 60
    plot_w = 600
    plot_h = 280
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    all_values = [value for values in series.values() for value in values]
    min_value = min(all_values) if all_values else 0.0
    max_value = max(all_values) if all_values else 0.0
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">{_svg_escape(title)}</text>',
        f'<line x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" y2="{plot_y + plot_h}" stroke="black"/>',
        f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_h}" stroke="black"/>',
        f'<text x="{plot_x + plot_w / 2}" y="{height - 28}" text-anchor="middle" font-family="sans-serif" font-size="13">{_svg_escape(x_label)}</text>',
        f'<text x="20" y="{plot_y + plot_h / 2}" text-anchor="middle" transform="rotate(-90 20 {plot_y + plot_h / 2})" font-family="sans-serif" font-size="13">{_svg_escape(y_label)}</text>',
        f'<text x="{plot_x - 8}" y="{plot_y + 4}" text-anchor="end" font-family="monospace" font-size="11">{max_value:.4g}</text>',
        f'<text x="{plot_x - 8}" y="{plot_y + plot_h}" text-anchor="end" font-family="monospace" font-size="11">{min_value:.4g}</text>',
    ]
    if not all_values:
        lines.append(f'<text x="{width / 2}" y="{height / 2}" text-anchor="middle" font-family="sans-serif" font-size="14">No data available</text>')
    for idx, (name, values) in enumerate(series.items()):
        color = colors[idx % len(colors)]
        points = _plot_points(values, plot_x, plot_y, plot_w, plot_h, min_value, max_value)
        if points:
            lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>')
        legend_y = plot_y + idx * 22
        lines.append(f'<rect x="{plot_x + plot_w + 18}" y="{legend_y - 10}" width="12" height="12" fill="{color}"/>')
        lines.append(f'<text x="{plot_x + plot_w + 36}" y="{legend_y}" font-family="sans-serif" font-size="12">{_svg_escape(name)}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_svg_bar_plot(
    path: Path,
    title: str,
    y_label: str,
    values: dict[str, float | None],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 760
    height = 440
    plot_x = 90
    plot_y = 60
    plot_w = 560
    plot_h = 280
    numeric_items = [(name, value) for name, value in values.items() if isinstance(value, (int, float))]
    max_value = max([abs(value) for _, value in numeric_items], default=1.0)
    scale = max(max_value, 1e-9)
    bar_w = plot_w / max(len(values), 1) * 0.55
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">{_svg_escape(title)}</text>',
        f'<line x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" y2="{plot_y + plot_h}" stroke="black"/>',
        f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_h}" stroke="black"/>',
        f'<text x="22" y="{plot_y + plot_h / 2}" text-anchor="middle" transform="rotate(-90 22 {plot_y + plot_h / 2})" font-family="sans-serif" font-size="13">{_svg_escape(y_label)}</text>',
    ]
    for idx, (name, value) in enumerate(values.items()):
        center_x = plot_x + (idx + 0.5) * (plot_w / max(len(values), 1))
        label = "n/a" if value is None else f"{float(value):.4g}"
        bar_h = 0.0 if value is None else (float(value) / scale) * plot_h
        x = center_x - bar_w / 2
        y = plot_y + plot_h - max(bar_h, 0)
        color = "#bdbdbd" if value is None else "#1f77b4"
        lines.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{max(bar_h, 0):.2f}" fill="{color}"/>')
        lines.append(f'<text x="{center_x:.2f}" y="{y - 6:.2f}" text-anchor="middle" font-family="monospace" font-size="11">{label}</text>')
        lines.append(f'<text x="{center_x:.2f}" y="{plot_y + plot_h + 18}" text-anchor="middle" font-family="sans-serif" font-size="11">{_svg_escape(name)}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def metric_series_from_trainer_logs(log_history: list[dict[str, Any]], metric_name: str) -> list[float]:
    values: list[float] = []
    for item in log_history:
        if not isinstance(item, dict):
            continue
        raw_value = item.get(metric_name)
        if raw_value is None and metric_name == "loss":
            raw_value = item.get("train_loss")
        value = _metric_or_none(raw_value)
        if value is not None:
            values.append(value)
    return values


def emit_observability_artifacts(output_dir: Path) -> dict[str, Any]:
    baseline_eval = load_json(output_dir / "baseline_model_eval.json")
    trained_eval = load_json(output_dir / "post_train_model_eval.json")
    summary = load_json(output_dir / "summary.json")
    trainer_log_path = output_dir / "trainer_log_history.json"
    trainer_logs = load_json(trainer_log_path) if trainer_log_path.exists() else []
    if not trainer_logs and isinstance(summary.get("trainer_metrics"), dict):
        trainer_logs = [summary["trainer_metrics"]]

    comparison = build_comparison(baseline_eval, trained_eval, summary)
    save_json(output_dir / "baseline_vs_trained_comparison.json", comparison)

    write_svg_bar_plot(
        output_dir / "reward_plot.svg",
        "Baseline vs Trained Mean Total Reward",
        "Mean total reward",
        {
            "baseline": comparison["comparison"]["mean_total_reward"]["baseline"],
            "trained": comparison["comparison"]["mean_total_reward"]["trained"],
        },
    )
    write_svg_line_plot(
        output_dir / "loss_plot.svg",
        "GRPO Training Loss",
        "Loss",
        {"train_loss": metric_series_from_trainer_logs(trainer_logs, "loss")},
        x_label="Logged train step",
    )
    write_svg_bar_plot(
        output_dir / "verifier_metrics_plot.svg",
        "Verifier / Rubric Metric Deltas",
        "Trained - baseline",
        {
            "Grounding F1": comparison["comparison"]["grounding_f1"]["delta"],
            "Drift Adapt": comparison["comparison"]["drift_adaptation_rate"]["delta"],
            "Search/Sub": comparison["comparison"]["search_to_submission_ratio"]["delta"],
            "Timeout Freq": comparison["comparison"]["timeout_frequency"]["delta"],
            "Schema Pass": comparison["comparison"]["schema_validation_pass_rate"]["delta"],
        },
    )
    return comparison


def require_training_stack(*, enable_unsloth_fast_rl: bool):
    try:
        import torch  # noqa: F401
        from datasets import Dataset  # noqa: F401
        from unsloth import FastLanguageModel  # noqa: F401

        if enable_unsloth_fast_rl:
            from unsloth import PatchFastRL  # noqa: F401

            PatchFastRL(FastLanguageModel)
        from trl import GRPOConfig, GRPOTrainer  # noqa: F401
    except Exception as exc:  # pragma: no cover - exercised only when optional deps are missing
        raise RuntimeError(
            "Optional training stack is missing. Install torch, datasets, accelerate, trl, and unsloth."
        ) from exc


def load_model_stack(model_name: str):
    import torch
    from unsloth import FastLanguageModel

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Unsloth GRPO smoke training requires CUDA. Use --skip-train for the local baseline smoke."
        )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    return model, tokenizer


def generate_action_text(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt")
    model_device = next(model.parameters()).device
    inputs = {name: tensor.to(model_device) for name, tensor in inputs.items()}

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    completion_ids = generated[0][prompt_len:]
    return tokenizer.decode(completion_ids, skip_special_tokens=True).strip()


def evaluate_model_policy(
    model,
    tokenizer,
    task_ids: list[str],
    episodes: int,
    max_episode_steps: int,
    max_new_tokens: int,
    label: str,
) -> dict[str, Any]:
    results = []

    for task_id in task_ids:
        for _ in range(episodes):
            def policy(prompt: str, observation) -> dict[str, Any]:
                completion = generate_action_text(model, tokenizer, build_prompt(prompt), max_new_tokens)
                action, used_fallback = parse_model_action(completion, observation, task_id)
                if used_fallback:
                    action["_fallback_used"] = True
                return action

            with RvedaTrainingBridge() as bridge:
                trace = bridge.rollout(policy, task_id=task_id, max_steps=max_episode_steps)
            results.append(serialize_rollout(task_id, trace))

    mean_reward = sum(item["total_reward"] for item in results) / max(len(results), 1)
    nonzero_reward_rate = sum(1 for item in results if item["total_reward"] != 0) / max(len(results), 1)
    return {
        "policy": label,
        "task_ids": task_ids,
        "episodes": results,
        "mean_total_reward": mean_reward,
        "rollout_summary": {
            "episode_count": len(results),
            "nonzero_reward_rate": nonzero_reward_rate,
            "mean_total_reward": mean_reward,
        },
    }


def run_grpo_smoke_train(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    enable_unsloth_fast_rl = not args.disable_unsloth_fast_rl
    require_training_stack(enable_unsloth_fast_rl=enable_unsloth_fast_rl)

    import torch
    from datasets import Dataset
    from unsloth import FastLanguageModel
    from trl import GRPOConfig, GRPOTrainer

    if enable_unsloth_fast_rl:
        from unsloth import PatchFastRL

        PatchFastRL(FastLanguageModel)

    train_rows = build_train_rows(args.task_ids, args.samples_per_task)
    save_json(output_dir / "train_rows_preview.json", train_rows[: min(4, len(train_rows))])
    train_dataset = Dataset.from_list(train_rows)

    model, tokenizer = load_model_stack(args.model_name)

    baseline_eval = evaluate_model_policy(
        model,
        tokenizer,
        task_ids=args.task_ids,
        episodes=args.episodes,
        max_episode_steps=args.max_episode_steps,
        max_new_tokens=args.max_new_tokens,
        label="pretrain_model",
    )
    save_json(output_dir / "baseline_model_eval.json", baseline_eval)

    train_args = GRPOConfig(
        output_dir=str(output_dir / "trainer_state"),
        max_steps=args.train_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-6,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        num_generations=2,
        max_completion_length=args.max_new_tokens,
        gradient_checkpointing=False,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        use_cpu=not torch.cuda.is_available(),
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=make_live_reward_func(),
        args=train_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    try:
        train_output = trainer.train()
    except TypeError as exc:
        if enable_unsloth_fast_rl and "grpo_accumulated_loss()" in str(exc):
            raise RuntimeError(
                "Unsloth FastRL patch is incompatible with the installed TRL/Unsloth build. "
                "Re-run with --disable-unsloth-fast-rl."
            ) from exc
        raise
    trainer_log_history = list(getattr(getattr(trainer, "state", None), "log_history", []) or [])
    save_json(output_dir / "trainer_log_history.json", trainer_log_history)
    trainer.save_model(str(output_dir / "model"))

    trained_eval = evaluate_model_policy(
        trainer.model,
        tokenizer,
        task_ids=args.task_ids,
        episodes=args.episodes,
        max_episode_steps=args.max_episode_steps,
        max_new_tokens=args.max_new_tokens,
        label="post_train_model",
    )
    save_json(output_dir / "post_train_model_eval.json", trained_eval)

    summary = {
        "model_name": args.model_name,
        "smoke_model": args.smoke_model,
        "task_ids": args.task_ids,
        "train_rows": len(train_rows),
        "train_steps": args.train_steps,
        "disable_unsloth_fast_rl": args.disable_unsloth_fast_rl,
        "baseline_mean_total_reward": baseline_eval["mean_total_reward"],
        "post_train_mean_total_reward": trained_eval["mean_total_reward"],
        "baseline_rollout_summary": baseline_eval["rollout_summary"],
        "post_train_rollout_summary": trained_eval["rollout_summary"],
        "trainer_metrics": dict(getattr(train_output, "metrics", {})),
        "saved_model_dir": str(output_dir / "model"),
    }
    save_json(output_dir / "summary.json", summary)
    emit_observability_artifacts(output_dir)
    return summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.emit_observability_only:
        comparison = emit_observability_artifacts(output_dir)
        print(json.dumps(comparison, indent=2))
        return

    save_json(output_dir / "command_metadata.json", build_command_metadata(args, output_dir))

    baseline = evaluate_scripted_policy(
        task_ids=args.task_ids,
        episodes=args.episodes,
        max_episode_steps=args.max_episode_steps,
    )
    save_json(output_dir / "scripted_baseline.json", baseline)

    if args.skip_train:
        save_json(
            output_dir / "summary.json",
            {
                "mode": "baseline_only",
                "model_name": args.model_name,
                "smoke_model": args.smoke_model,
                "task_ids": args.task_ids,
                "disable_unsloth_fast_rl": args.disable_unsloth_fast_rl,
                "baseline_mean_total_reward": baseline["mean_total_reward"],
                "baseline_rollout_summary": baseline["rollout_summary"],
                "saved_files": ["command_metadata.json", "scripted_baseline.json"],
            },
        )
        print(f"Saved baseline smoke artifacts to {output_dir}")
        return

    try:
        summary = run_grpo_smoke_train(args, output_dir)
    except Exception as exc:
        save_failure_status(output_dir, exc)
        save_failed_attempt_summary(output_dir, args, baseline, exc)
        raise
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
