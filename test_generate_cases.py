"""Tests for Sprint 3 synthetic V2 case generation."""

from __future__ import annotations

import json
from pathlib import Path

from rveda.generate_cases import MODULE_KEYS, generate_cases, write_generated_cases
from rveda.server.rveda_environment import RvedaEnvironment


def test_generate_cases_emits_expected_supervised_fields() -> None:
    tasks = generate_cases(seed=7)

    assert len(tasks) == 12
    assert len({task["task_id"] for task in tasks}) == len(tasks)

    for task in tasks:
        assert task["difficulty"] in {"easy", "medium", "hard"}
        assert task["split"] in {"train", "eval", "smoke"}
        assert set(task["ehr_modules"]) == set(MODULE_KEYS)
        assert task["target_evidence_ids"] == task["target_evidence"]
        assert task["split_tags"]["split"] == task["split"]
        assert task["split_tags"]["difficulty"] == task["difficulty"]
        assert task["split_tags"]["schema_version"] == task["schema_validation_expectations"]["required_schema_version"]
        assert "terminal_answer_labels" in task
        assert "schema_validation_expectations" in task
        assert "verification_checkpoints" in task
        assert "generator_metadata" in task


def test_target_evidence_ids_exist_once_and_are_required_for_grounding() -> None:
    tasks = generate_cases(seed=7)

    for task in tasks:
        evidence_index = {}
        for module in task["ehr_modules"].values():
            for evidence in module["evidence"]:
                evidence_index.setdefault(evidence["evidence_id"], []).append(evidence)

        for evidence_id in task["target_evidence_ids"]:
            matches = evidence_index.get(evidence_id, [])
            assert len(matches) == 1
            assert matches[0]["required_for_grounding"] is True
            assert task["target_code"] in matches[0]["supports_codes"]


def test_drift_and_schema_expectations_are_consistent() -> None:
    tasks = generate_cases(seed=7)

    drift_enabled = [task for task in tasks if task["drift"]["enabled"]]
    drift_disabled = [task for task in tasks if not task["drift"]["enabled"]]

    assert drift_enabled
    assert drift_disabled

    for task in drift_enabled:
        drift = task["drift"]
        schema_expectations = task["schema_validation_expectations"]
        valid_claim = schema_expectations["expected_valid_claim"]
        invalid_examples = schema_expectations["expected_invalid_claims"]

        assert drift["trigger_step"] is not None
        assert drift["to_schema_version"] == schema_expectations["required_schema_version"]
        assert drift["required_fields"] == schema_expectations["required_fields"]
        assert drift["expected_adaptation_action"] == "VALIDATE_CLAIM_SCHEMA"
        for field_name in schema_expectations["required_fields"]:
            assert field_name in valid_claim
        assert any(example["label"] == "wrong_schema_version" for example in invalid_examples)
        assert any(example["label"] == "missing_policy_attestations" for example in invalid_examples)

    for task in drift_disabled:
        drift = task["drift"]
        schema_expectations = task["schema_validation_expectations"]

        assert drift["trigger_step"] is None
        assert drift["expected_adaptation_action"] is None
        assert task["claim_schema_version"] == schema_expectations["required_schema_version"]


def test_generate_cases_are_current_runtime_trainable() -> None:
    tasks = generate_cases(seed=7)

    for task in tasks:
        runtime_path = task["current_runtime_path"]
        assert runtime_path["supported_actions"] == [
            "QUERY_EHR",
            "SEARCH",
            "DETAILS",
            "CHECK_POLICY",
            "VALIDATE_CLAIM_SCHEMA",
            "REASONING_LOG",
            "SUBMIT",
        ]
        assert runtime_path["recommended_ehr_modules"]
        assert runtime_path["recommended_ehr_queries"]
        assert runtime_path["recommended_search_queries"]
        assert runtime_path["expected_submit_code"] == task["target_code"]


def test_write_generated_cases_emits_loader_ready_files(tmp_path: Path) -> None:
    written = write_generated_cases(output_dir=tmp_path, seed=7)
    paths = sorted(tmp_path.glob("v2_task_*.json"))

    assert len(paths) == len(written)
    assert (tmp_path / "synthetic_cases_manifest.json").exists()

    loaded = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    assert {task["task_id"] for task in loaded} == {task["task_id"] for task in written}


def test_generated_task_pack_is_visible_to_environment_loader() -> None:
    env = RvedaEnvironment()

    loaded_task_ids = {task["task_id"] for task in env._v2_tasks}

    assert "v2_easy_overweight_schema_v1" in loaded_task_ids
    assert any(task_id.startswith("v2_train_") or task_id.startswith("v2_task_train_") for task_id in loaded_task_ids)
