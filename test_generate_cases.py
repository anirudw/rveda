from __future__ import annotations

import json
from pathlib import Path

from generate_cases import generate_cases, write_generated_cases


def test_generate_cases_are_deterministic_and_training_ready(tmp_path: Path) -> None:
    first = generate_cases(seed=7)
    second = generate_cases(seed=7)

    assert first == second
    assert len(first) > 1
    assert any(task["split"] == "train" for task in first)
    assert any(task.get("drift", {}).get("enabled") for task in first)

    sample = first[0]
    assert sample["task_id"].startswith("v2_task_")
    assert sample["target_evidence_ids"]
    assert sample["target_evidence"] == sample["target_evidence_ids"]
    assert sample["ehr_modules"]
    assert sample["schema_validation_expectations"]["expected_valid_claim"]
    assert sample["verification_checkpoints"]

    written = write_generated_cases(output_dir=tmp_path, seed=7)
    paths = sorted(tmp_path.glob("v2_task_*.json"))
    assert len(paths) == len(written)
    assert (tmp_path / "synthetic_cases_manifest.json").exists()

    loaded = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    assert {task["task_id"] for task in loaded} == {task["task_id"] for task in written}
