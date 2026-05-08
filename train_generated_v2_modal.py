"""Modal entrypoint for rerunning the generated V2 GRPO training path."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

from train_generated_v2_modal_config import (
    PRESETS,
    build_emit_observability_command,
    build_generate_cases_command,
    build_train_command,
    expected_artifact_paths,
    get_preset,
)


ARTIFACT_VOLUME_NAME = "rveda-training-artifacts"
REMOTE_REPO_PATH = Path("/workspace/rveda")
REMOTE_VOLUME_PATH = Path("/modal_artifacts")

try:
    import modal
except ImportError:  # Modal is optional for local tests and normal repo validation.
    modal = None


def _run(command: list[str], *, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _zip_dir(source_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source_dir.parent))


def _verify_artifacts(output_dir: Path) -> list[str]:
    missing = [str(path) for path in expected_artifact_paths(output_dir) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing expected artifacts: " + ", ".join(missing))
    return [str(path) for path in expected_artifact_paths(output_dir)]


def run_training_path(
    *,
    preset_name: str,
    output_tag: str | None = None,
    clean_output: bool = False,
    run_pytest: bool | None = None,
    repo_dir: Path | None = None,
    zip_destination: Path | None = None,
) -> dict[str, Any]:
    repo_root = repo_dir or Path.cwd()
    preset = get_preset(preset_name)
    output_dir = Path("artifacts") / output_tag if output_tag else preset.output_dir
    output_dir = repo_root / output_dir

    if output_dir.exists():
        if not clean_output:
            raise FileExistsError(f"{output_dir} already exists. Pass --clean-output true to replace it.")
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    should_run_pytest = preset.run_pytest if run_pytest is None else run_pytest
    if should_run_pytest:
        _run([sys.executable, "-m", "pytest", "-q"], cwd=repo_root)

    if preset.regenerate_cases:
        _run(build_generate_cases_command(python_executable=sys.executable), cwd=repo_root)

    _run(
        build_train_command(
            preset,
            output_dir=output_dir.relative_to(repo_root),
            python_executable=sys.executable,
        ),
        cwd=repo_root,
    )
    _run(
        build_emit_observability_command(
            output_dir.relative_to(repo_root),
            python_executable=sys.executable,
        ),
        cwd=repo_root,
    )
    verified = _verify_artifacts(output_dir)

    zip_path = zip_destination or output_dir.with_suffix(".zip")
    _zip_dir(output_dir, zip_path)
    return {
        "preset": preset_name,
        "output_dir": str(output_dir),
        "zip_path": str(zip_path),
        "verified_artifacts": verified,
    }


if modal is not None:
    app = modal.App("rveda-generated-v2-grpo")
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git")
        .pip_install(
            "accelerate",
            "datasets",
            "peft",
            "pytest",
            "sentencepiece",
            "torch",
            "transformers",
            "trl",
            "uv",
        )
        .add_local_dir(".", remote_path=str(REMOTE_REPO_PATH))
    )
    artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=True)

    @app.function(
        image=image,
        gpu=os.getenv("RVEDA_MODAL_GPU", "T4"),
        timeout=60 * 60 * 6,
        volumes={str(REMOTE_VOLUME_PATH): artifact_volume},
    )
    def run_training_remote(
        preset: str,
        output_tag: str,
        clean_output: bool = False,
        run_pytest: bool = True,
    ) -> dict[str, Any]:
        zip_destination = REMOTE_VOLUME_PATH / f"{output_tag}.zip"
        result = run_training_path(
            preset_name=preset,
            output_tag=output_tag,
            clean_output=clean_output,
            run_pytest=run_pytest,
            repo_dir=REMOTE_REPO_PATH,
            zip_destination=zip_destination,
        )
        artifact_volume.commit()
        result["download_command"] = (
            f"modal volume get {ARTIFACT_VOLUME_NAME} /{output_tag}.zip artifacts/modal_downloads"
        )
        return result

    @app.local_entrypoint()
    def modal_main(
        preset: str = "phase3_1p5b_presentable",
        output_tag: str = "phase3_modal_1p5b",
        clean_output: bool = False,
        run_pytest: bool = True,
    ) -> None:
        print(run_training_remote.remote(preset, output_tag, clean_output, run_pytest))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run generated V2 GRPO training locally.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="phase3_1p5b_presentable")
    parser.add_argument("--output-tag", default=None)
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--skip-pytest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(
        run_training_path(
            preset_name=args.preset,
            output_tag=args.output_tag,
            clean_output=args.clean_output,
            run_pytest=not args.skip_pytest,
        )
    )


if __name__ == "__main__":
    main()
