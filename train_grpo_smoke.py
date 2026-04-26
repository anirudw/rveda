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
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from trl_bridge import BridgeStep, RolloutTrace, RvedaTrainingBridge


DEFAULT_TASK_IDS = ["easy_endo_1"]
ACTION_TYPES = {"SEARCH", "DETAILS", "SUBMIT", "QUERY_EHR"}
ACTION_INSTRUCTION = (
    "Return exactly one JSON object with keys action_type, query, and optional "
    "module. Valid action_type values are SEARCH, DETAILS, SUBMIT, QUERY_EHR. "
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
    parser.add_argument("--output-dir", default="artifacts/grpo_smoke", help="Directory for JSON outputs.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct", help="Base model for smoke GRPO.")
    parser.add_argument("--task-ids", nargs="+", default=DEFAULT_TASK_IDS, help="Task ids used for smoke train/eval.")
    parser.add_argument("--samples-per-task", type=int, default=4, help="Number of prompt rows per task.")
    parser.add_argument("--episodes", type=int, default=1, help="Evaluation episodes per task.")
    parser.add_argument("--max-episode-steps", type=int, default=3, help="Max steps during smoke evaluation.")
    parser.add_argument("--train-steps", type=int, default=2, help="Max GRPO optimization steps.")
    parser.add_argument("--max-new-tokens", type=int, default=96, help="Generation cap for action JSON.")
    parser.add_argument("--skip-train", action="store_true", help="Only run the live env baseline smoke.")
    return parser.parse_args()


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


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


def default_action_for_observation(observation, task_id: str) -> dict[str, Any]:
    if observation.ehr_map:
        for module_name, module_state in observation.ehr_map.items():
            if module_state.query_budget_remaining > 0 and module_state.revealed_count == 0:
                return {
                    "action_type": "QUERY_EHR",
                    "module": module_name,
                    "query": SAFE_EHR_QUERY_BY_TASK.get(task_id, "BMI weight"),
                }

    if observation.search_results:
        if observation.grading.action_type == "SEARCH":
            return {
                "action_type": "DETAILS",
                "query": observation.search_results[0].code,
            }
        return {
            "action_type": "SUBMIT",
            "query": observation.search_results[0].code,
        }

    return {
        "action_type": "SEARCH",
        "query": SAFE_SEARCH_BY_TASK.get(task_id, "weight"),
    }


def coerce_action_dict(raw_action: dict[str, Any], observation, task_id: str) -> tuple[dict[str, Any], bool]:
    action_type = str(raw_action.get("action_type", "")).strip().upper()
    query = str(raw_action.get("query", "")).strip()
    module = raw_action.get("module")
    module_text = str(module).strip() if module is not None else None

    if action_type not in ACTION_TYPES or not query:
        return default_action_for_observation(observation, task_id), True
    if action_type == "QUERY_EHR" and not module_text:
        return default_action_for_observation(observation, task_id), True

    action = {
        "action_type": action_type,
        "query": query,
    }
    if module_text is not None:
        action["module"] = module_text
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


def require_training_stack():
    try:
        import torch  # noqa: F401
        from datasets import Dataset  # noqa: F401
        from unsloth import FastLanguageModel  # noqa: F401
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
    require_training_stack()

    import torch
    from datasets import Dataset
    from unsloth import FastLanguageModel, PatchFastRL
    from trl import GRPOConfig, GRPOTrainer

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
    train_output = trainer.train()
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
        "task_ids": args.task_ids,
        "train_rows": len(train_rows),
        "train_steps": args.train_steps,
        "baseline_mean_total_reward": baseline_eval["mean_total_reward"],
        "post_train_mean_total_reward": trained_eval["mean_total_reward"],
        "baseline_rollout_summary": baseline_eval["rollout_summary"],
        "post_train_rollout_summary": trained_eval["rollout_summary"],
        "trainer_metrics": dict(getattr(train_output, "metrics", {})),
        "saved_model_dir": str(output_dir / "model"),
    }
    save_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
                "task_ids": args.task_ids,
                "baseline_mean_total_reward": baseline["mean_total_reward"],
                "baseline_rollout_summary": baseline["rollout_summary"],
                "saved_files": ["scripted_baseline.json"],
            },
        )
        print(f"Saved baseline smoke artifacts to {output_dir}")
        return

    summary = run_grpo_smoke_train(args, output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
