from pathlib import Path

import pytest

import train_generated_v2_modal
from train_generated_v2_modal_config import (
    PHASE3_TRAIN_TASK_IDS,
    build_emit_observability_command,
    build_train_command,
    expected_artifact_paths,
    get_preset,
)


def test_phase3_preset_matches_notebook_contract() -> None:
    preset = get_preset("phase3_1p5b_presentable")

    assert preset.model_name == "Qwen/Qwen2.5-1.5B-Instruct"
    assert preset.task_ids == PHASE3_TRAIN_TASK_IDS
    assert preset.samples_per_task == 2
    assert preset.episodes == 2
    assert preset.train_steps == 8
    assert preset.max_episode_steps == 8
    assert preset.disable_unsloth_fast_rl is True


def test_build_train_command_contains_required_flags() -> None:
    preset = get_preset("phase3_1p5b_presentable")
    command = build_train_command(preset, output_dir=Path("artifacts/example"), python_executable="py")

    assert command[:2] == ["py", "train_grpo_smoke.py"]
    assert command[command.index("--output-dir") + 1] == "artifacts\\example" or command[
        command.index("--output-dir") + 1
    ] == "artifacts/example"
    assert "--disable-unsloth-fast-rl" in command
    for task_id in PHASE3_TRAIN_TASK_IDS:
        assert task_id in command


def test_observability_command_and_expected_artifacts_share_output_dir() -> None:
    output_dir = Path("artifacts/example")

    command = build_emit_observability_command(output_dir, python_executable="py")
    artifacts = expected_artifact_paths(output_dir)

    assert command == ["py", "train_grpo_smoke.py", "--output-dir", str(output_dir), "--emit-observability-only"]
    assert output_dir / "summary.json" in artifacts
    assert output_dir / "comparison_summary.json" in artifacts


def test_unknown_preset_error_lists_available_presets() -> None:
    with pytest.raises(ValueError, match="phase3_1p5b_presentable"):
        get_preset("missing")

    assert callable(train_generated_v2_modal.run_training_path)
