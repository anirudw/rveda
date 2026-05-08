"""Configuration helpers for the generated V2 Modal training runner."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path


QWEN25_1P5B_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

PHASE3_TRAIN_TASK_IDS = (
    "v2_task_train_easy_hypothyroid_unspecified_v1_001",
    "v2_task_train_easy_overweight_counseling_v1_002",
    "v2_task_train_medium_chest_pain_observation_v1_003",
    "v2_task_train_medium_diabetes_hyperglycemia_drift_drift_004",
)


@dataclass(frozen=True)
class TrainingPreset:
    name: str
    model_name: str
    task_ids: tuple[str, ...]
    samples_per_task: int
    episodes: int
    train_steps: int
    max_episode_steps: int
    max_new_tokens: int
    disable_unsloth_fast_rl: bool
    output_dir: Path
    run_pytest: bool = True
    regenerate_cases: bool = True

    def with_output_dir(self, output_dir: Path) -> "TrainingPreset":
        return replace(self, output_dir=output_dir)


PRESETS: dict[str, TrainingPreset] = {
    "phase3_1p5b_presentable": TrainingPreset(
        name="phase3_1p5b_presentable",
        model_name=QWEN25_1P5B_MODEL,
        task_ids=PHASE3_TRAIN_TASK_IDS,
        samples_per_task=2,
        episodes=2,
        train_steps=8,
        max_episode_steps=8,
        max_new_tokens=96,
        disable_unsloth_fast_rl=True,
        output_dir=Path("artifacts/grpo_phase3_1p5b_presentable"),
    ),
    "tiny_1p5b_plain_trl": TrainingPreset(
        name="tiny_1p5b_plain_trl",
        model_name=QWEN25_1P5B_MODEL,
        task_ids=(PHASE3_TRAIN_TASK_IDS[0],),
        samples_per_task=1,
        episodes=1,
        train_steps=1,
        max_episode_steps=4,
        max_new_tokens=64,
        disable_unsloth_fast_rl=True,
        output_dir=Path("artifacts/grpo_tiny_1p5b_plain_trl"),
    ),
    "small_1p5b_plain_trl": TrainingPreset(
        name="small_1p5b_plain_trl",
        model_name=QWEN25_1P5B_MODEL,
        task_ids=PHASE3_TRAIN_TASK_IDS,
        samples_per_task=1,
        episodes=1,
        train_steps=2,
        max_episode_steps=6,
        max_new_tokens=96,
        disable_unsloth_fast_rl=True,
        output_dir=Path("artifacts/grpo_small_1p5b_plain_trl"),
    ),
    "small_1p5b_submit_check": TrainingPreset(
        name="small_1p5b_submit_check",
        model_name=QWEN25_1P5B_MODEL,
        task_ids=(PHASE3_TRAIN_TASK_IDS[1],),
        samples_per_task=2,
        episodes=2,
        train_steps=4,
        max_episode_steps=8,
        max_new_tokens=96,
        disable_unsloth_fast_rl=True,
        output_dir=Path("artifacts/grpo_small_1p5b_submit_check"),
    ),
}


REQUIRED_JSON_ARTIFACTS = (
    "command_metadata.json",
    "scripted_baseline.json",
    "train_rows_preview.json",
    "baseline_model_eval.json",
    "post_train_model_eval.json",
    "trainer_log_history.json",
    "summary.json",
    "comparison_summary.json",
)


def get_preset(name: str) -> TrainingPreset:
    try:
        return PRESETS[name]
    except KeyError as exc:
        available = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown preset {name!r}. Available presets: {available}") from exc


def build_train_command(
    preset: TrainingPreset,
    *,
    output_dir: Path | None = None,
    python_executable: str = "python",
) -> list[str]:
    resolved_output_dir = output_dir or preset.output_dir
    command = [
        python_executable,
        "train_grpo_smoke.py",
        "--model-name",
        preset.model_name,
        "--output-dir",
        str(resolved_output_dir),
        "--task-ids",
        *preset.task_ids,
        "--samples-per-task",
        str(preset.samples_per_task),
        "--episodes",
        str(preset.episodes),
        "--train-steps",
        str(preset.train_steps),
        "--max-episode-steps",
        str(preset.max_episode_steps),
        "--max-new-tokens",
        str(preset.max_new_tokens),
    ]
    if preset.disable_unsloth_fast_rl:
        command.append("--disable-unsloth-fast-rl")
    return command


def build_emit_observability_command(
    output_dir: Path,
    *,
    python_executable: str = "python",
) -> list[str]:
    return [
        python_executable,
        "train_grpo_smoke.py",
        "--output-dir",
        str(output_dir),
        "--emit-observability-only",
    ]


def build_generate_cases_command(*, python_executable: str = "python") -> list[str]:
    return [
        python_executable,
        "generate_cases.py",
        "--output-dir",
        "examples",
        "--output",
        "examples/generated_v2_tasks.json",
        "--seed",
        "7",
        "--clean",
    ]


def expected_artifact_paths(output_dir: Path) -> list[Path]:
    return [output_dir / name for name in REQUIRED_JSON_ARTIFACTS]
