"""Minimal smoke test for the training bridge path."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from rveda.trl_bridge import RvedaTrainingBridge
from rveda import train_grpo_smoke


V2_TASK_ID = "v2_easy_overweight_schema_v1"


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _baseline_policy(_prompt: str, _observation) -> dict[str, str]:
    return {"action_type": "SEARCH", "query": "weight"}


def _smoke_trained_policy(_prompt: str, observation) -> dict[str, str]:
    if observation.ehr_map:
        for module_name, module_state in observation.ehr_map.items():
            if module_state.query_budget_remaining > 0:
                return {
                    "action_type": "QUERY_EHR",
                    "module": module_name,
                    "query": "BMI",
                }
    return {"action_type": "SEARCH", "query": "weight"}


def test_training_bridge_smoke_rollouts_are_distinguishable(tmp_path: Path) -> None:
    with RvedaTrainingBridge() as bridge:
        baseline_trace = bridge.rollout(
            _baseline_policy,
            task_id=V2_TASK_ID,
            max_steps=2,
        )

    with RvedaTrainingBridge() as bridge:
        smoke_trained_trace = bridge.rollout(
            _smoke_trained_policy,
            task_id=V2_TASK_ID,
            max_steps=2,
        )

    baseline_payload = {
        "policy": "baseline",
        "task_id": V2_TASK_ID,
        "total_reward": baseline_trace.total_reward,
        "done": baseline_trace.done,
        "steps": [step.action for step in baseline_trace.steps],
    }
    smoke_trained_payload = {
        "policy": "smoke_trained",
        "task_id": V2_TASK_ID,
        "total_reward": smoke_trained_trace.total_reward,
        "done": smoke_trained_trace.done,
        "steps": [step.action for step in smoke_trained_trace.steps],
    }

    baseline_path = tmp_path / "baseline_rollout.json"
    smoke_trained_path = tmp_path / "smoke_trained_rollout.json"
    _save_json(baseline_path, baseline_payload)
    _save_json(smoke_trained_path, smoke_trained_payload)

    baseline_saved = json.loads(baseline_path.read_text(encoding="utf-8"))
    smoke_trained_saved = json.loads(smoke_trained_path.read_text(encoding="utf-8"))

    assert baseline_saved["policy"] == "baseline"
    assert smoke_trained_saved["policy"] == "smoke_trained"
    assert baseline_saved["task_id"] == V2_TASK_ID
    assert smoke_trained_saved["task_id"] == V2_TASK_ID
    assert baseline_saved["steps"]
    assert smoke_trained_saved["steps"]
    assert baseline_saved != smoke_trained_saved


def test_small_smoke_model_preset_prefers_1p5b_and_output_dir(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_grpo_smoke.py", "--smoke-model", "qwen2.5-1.5b"],
    )

    args = train_grpo_smoke.parse_args()

    assert args.model_name == train_grpo_smoke.QWEN25_1P5B_MODEL
    assert args.output_dir == train_grpo_smoke.QWEN25_1P5B_OUTPUT_DIR


def _rollout_step(
    action_type: str,
    *,
    last_error: str | None = None,
    invalid_reason: str | None = None,
    reward_metrics: dict | None = None,
    drift_notice: dict | None = None,
) -> dict:
    return {
        "action": {"action_type": action_type},
        "reward": 0.0,
        "done": action_type == "SUBMIT",
        "info": {"v2_verifier_snapshot": {}},
        "observation": {
            "last_error": last_error,
            "invalid_reason": invalid_reason,
            "reward_metrics": reward_metrics or {},
            "drift_notice": drift_notice,
            "metadata": {},
        },
    }


def test_policy_metrics_counts_schema_validation_pass_rate() -> None:
    payload = {
        "episodes": [
            {
                "total_reward": 0.5,
                "steps": [
                    _rollout_step("CHECK_POLICY"),
                    _rollout_step(
                        "VALIDATE_CLAIM_SCHEMA",
                        reward_metrics={"schema_compliance": 1.0},
                    ),
                    _rollout_step(
                        "VALIDATE_CLAIM_SCHEMA",
                        last_error="schema_validation_failed",
                        invalid_reason="Missing required fields: evidence_ids",
                        reward_metrics={"schema_compliance": 0.0},
                    ),
                ],
            }
        ]
    }

    metrics = train_grpo_smoke.policy_metrics(payload)

    assert metrics["action_counts"]["CHECK_POLICY"] == 1
    assert metrics["action_counts"]["VALIDATE_CLAIM_SCHEMA"] == 2
    assert metrics["schema_validation_attempts"] == 2
    assert metrics["schema_validation_passes"] == 1
    assert metrics["schema_validation_pass_rate"] == 0.5
    assert "VALIDATE_CLAIM_SCHEMA pass rate" in metrics["schema_validation_pass_rate_note"]


def test_policy_metrics_counts_post_drift_adapted_submit_rate() -> None:
    drift_notice = {
        "active": True,
        "trigger_step": 1,
        "from_schema_version": "v1",
        "to_schema_version": "v2",
    }
    payload = {
        "episodes": [
            {
                "total_reward": 0.99,
                "steps": [
                    _rollout_step("CHECK_POLICY", drift_notice=drift_notice),
                    _rollout_step(
                        "VALIDATE_CLAIM_SCHEMA",
                        drift_notice=drift_notice,
                        reward_metrics={
                            "schema_compliance": 1.0,
                            "drift_adaptation": 0.75,
                        },
                    ),
                    _rollout_step(
                        "SUBMIT",
                        drift_notice=drift_notice,
                        reward_metrics={"drift_adaptation": 1.0},
                    ),
                ],
            },
            {
                "total_reward": 0.4,
                "steps": [
                    _rollout_step("CHECK_POLICY", drift_notice=drift_notice),
                    _rollout_step(
                        "SUBMIT",
                        drift_notice=drift_notice,
                        reward_metrics={"drift_adaptation": 0.0},
                    ),
                ],
            },
        ]
    }

    metrics = train_grpo_smoke.policy_metrics(payload)

    assert metrics["drift_triggered_episodes"] == 2
    assert metrics["drift_adapted_episodes"] == 1
    assert metrics["drift_adaptation_rate"] == 0.5
    assert "Post-drift episode pass rate" in metrics["drift_adaptation_rate_note"]


def test_policy_metrics_keep_unavailable_rates_when_no_denominator() -> None:
    payload = {
        "episodes": [
            {
                "total_reward": 0.1,
                "steps": [
                    _rollout_step("SEARCH"),
                    _rollout_step("DETAILS"),
                ],
            }
        ]
    }

    metrics = train_grpo_smoke.policy_metrics(payload)

    assert metrics["schema_validation_attempts"] == 0
    assert metrics["schema_validation_pass_rate"] is None
    assert metrics["drift_triggered_episodes"] == 0
    assert metrics["drift_adaptation_rate"] is None


def test_generated_task_metadata_guides_scripted_policy() -> None:
    task_id = "v2_train_medium_diabetes_hyperglycemia_drift_drift_004"
    context = train_grpo_smoke._task_hint(task_id)

    search_queries = train_grpo_smoke._recommended_search_queries(context)
    ehr_modules = train_grpo_smoke._recommended_ehr_modules(context)

    assert search_queries[0] == "Type 2 diabetes mellitus with hyperglycemia"
    assert "labs" in ehr_modules
    assert "medications" in ehr_modules
    assert train_grpo_smoke._ehr_query_for_module(context, "labs") == "ev_diabetes_hyperglycemia_drift_01"
    assert train_grpo_smoke._ehr_query_for_module(context, "medications") == "ev_diabetes_hyperglycemia_drift_02"


def test_emit_observability_writes_comparison_summary_with_real_rates(tmp_path: Path) -> None:
    drift_notice = {
        "active": True,
        "trigger_step": 1,
        "from_schema_version": "v1",
        "to_schema_version": "v2",
    }
    baseline_eval = {
        "policy": "baseline",
        "task_ids": ["v2_drift_case"],
        "mean_total_reward": 0.1,
        "episodes": [
            {
                "total_reward": 0.1,
                "steps": [
                    _rollout_step("CHECK_POLICY", drift_notice=drift_notice),
                    _rollout_step(
                        "VALIDATE_CLAIM_SCHEMA",
                        drift_notice=drift_notice,
                        last_error="schema_validation_failed",
                        invalid_reason="Missing required fields: policy_attestations",
                        reward_metrics={"schema_compliance": 0.0},
                    ),
                    _rollout_step(
                        "SUBMIT",
                        drift_notice=drift_notice,
                        reward_metrics={"drift_adaptation": 0.0},
                    ),
                ],
            }
        ],
    }
    trained_eval = {
        "policy": "trained",
        "task_ids": ["v2_drift_case"],
        "mean_total_reward": 0.9,
        "episodes": [
            {
                "total_reward": 0.9,
                "steps": [
                    _rollout_step("CHECK_POLICY", drift_notice=drift_notice),
                    _rollout_step(
                        "VALIDATE_CLAIM_SCHEMA",
                        drift_notice=drift_notice,
                        reward_metrics={
                            "schema_compliance": 1.0,
                            "drift_adaptation": 0.75,
                        },
                    ),
                    _rollout_step(
                        "SUBMIT",
                        drift_notice=drift_notice,
                        reward_metrics={"drift_adaptation": 1.0},
                    ),
                ],
            }
        ],
    }
    summary = {
        "model_name": "unit-test-model",
        "task_ids": ["v2_drift_case"],
        "train_steps": 1,
        "trainer_metrics": {"train_loss": 0.1},
    }

    _save_json(tmp_path / "baseline_model_eval.json", baseline_eval)
    _save_json(tmp_path / "post_train_model_eval.json", trained_eval)
    _save_json(tmp_path / "summary.json", summary)

    comparison = train_grpo_smoke.emit_observability_artifacts(tmp_path)
    summary_payload = json.loads((tmp_path / "comparison_summary.json").read_text(encoding="utf-8"))

    assert (tmp_path / "baseline_vs_trained_comparison.json").exists()
    assert (tmp_path / "comparison_summary.json").exists()
    assert comparison["trained_policy"]["schema_validation_pass_rate"] == 1.0
    assert comparison["trained_policy"]["drift_adaptation_rate"] == 1.0
    assert summary_payload["comparison"]["schema_validation_pass_rate"]["delta"] == 1.0
    assert summary_payload["comparison"]["drift_adaptation_rate"]["delta"] == 1.0


def main() -> None:
    tmp_path = Path(tempfile.mkdtemp(prefix="rveda_training_smoke_"))
    test_training_bridge_smoke_rollouts_are_distinguishable(tmp_path)
    print(f"Training smoke test passed: {tmp_path}")


if __name__ == "__main__":
    main()
