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


def main() -> None:
    tmp_path = Path(tempfile.mkdtemp(prefix="rveda_training_smoke_"))
    test_training_bridge_smoke_rollouts_are_distinguishable(tmp_path)
    print(f"Training smoke test passed: {tmp_path}")


if __name__ == "__main__":
    main()
