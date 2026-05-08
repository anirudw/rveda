# Modal Training

This project now has a Modal runner for the generated V2 GRPO flow at `train_generated_v2_modal.py`.

It does not replace or delete the Colab notebooks. The Modal path reuses the same core script, `train_grpo_smoke.py`, and keeps the same generated-case flow, preflight checks, observability regeneration, and required artifact contract that the notebook checks.

## Supported target

The primary supported preset is `phase3_1p5b_presentable`, because that is the strongest current evidence path in the repo:

- model: `Qwen/Qwen2.5-1.5B-Instruct`
- train tasks: the same four Phase 3 task ids used in the notebook
- trainer path: plain TRL with `--disable-unsloth-fast-rl`
- expected outputs: the same JSON and SVG artifacts the notebook asserts

The runner also supports:

- `tiny_1p5b_plain_trl`
- `small_1p5b_plain_trl`
- `small_1p5b_submit_check`

## Local prerequisites

Install the local Modal CLI and authenticate:

```powershell
pip install modal
modal setup
```

Optional:

- set `HF_TOKEN` if you want authenticated Hugging Face downloads
- set `RVEDA_MODAL_GPU=L4` if you want to override the default `T4`

## Start a Phase 3 run

From the repo root:

```powershell
modal run train_generated_v2_modal.py --preset phase3_1p5b_presentable --output-tag phase3_modal_1p5b
```

What this does remotely:

1. installs the repo plus training dependencies into a Modal image
2. optionally runs `pytest -q`
3. regenerates deterministic V2 cases with seed `7`
4. runs the same easy/medium/hard preflight rollouts as the notebook
5. launches `train_grpo_smoke.py` with the notebook-equivalent preset
6. reruns `--emit-observability-only`
7. verifies the notebook-required artifacts
8. writes a zip bundle into the Modal artifacts volume

## Download artifacts

The run result prints a `download_command`. It will look like this:

```powershell
modal volume get rveda-training-artifacts /phase3_modal_1p5b.zip artifacts/modal_downloads
```

You can also inspect the volume directly:

```powershell
modal volume ls rveda-training-artifacts /
```

## Notes

- Exact reward numbers may vary slightly across runs and hardware, but the Modal path uses the same training script and Phase 3 preset as the notebook, so the pipeline and artifact shape are the same.
- The Modal runner avoids the notebook's Colab-only download cells and destructive repo reset behavior.
- If you rerun the same `output-tag`, the runner will stop instead of overwriting artifacts unless you pass `--clean-output true`.
