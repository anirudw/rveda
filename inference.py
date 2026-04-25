"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each tasks should return score in [0, 1]

  Example:
    [START] task=click-test env=miniwob model=Qwen3-VL-30B
    [STEP] step=1 action=click('123') reward=0.00 done=false error=null
    [STEP] step=2 action=fill('456','text') reward=0.00 done=false error=null
    [STEP] step=3 action=click('789') reward=1.00 done=true error=null
    [END] success=true steps=3 score=1.00 rewards=0.00,0.00,1.00
"""

import asyncio
import json
import os
import textwrap
from pathlib import Path
from typing import Any, List, Optional

#dot_env
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

from models import MedicalAction as RvedaAction
from client import RvedaEnv

IMAGE_NAME = os.getenv("IMAGE_NAME") # If you are using docker image
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = os.getenv("RVEDA_BENCHMARK", "rveda")
MAX_STEPS = 8
TEMPERATURE = 0.7
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.1  # normalized score in [0, 1]
MAX_TOTAL_REWARD = 1.0
OPEN_SCORE_MIN = 0.001
OPEN_SCORE_MAX = 0.999
VALID_ACTION_TYPES = {"SEARCH", "DETAILS", "SUBMIT", "QUERY_EHR"}

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert Medical Coder agent.
    Your task is to review a patient note, reveal any hidden EHR evidence when needed, search the ICD-10 taxonomy, and SUBMIT the most accurate code.
    
    You must reply ONLY with a valid JSON object matching this schema:
    {"action_type": "SEARCH", "query": "keyword"} - Use short, single-word keywords (e.g., "hypertension").
    {"action_type": "DETAILS", "query": "code"} - Look up rules for a specific code (e.g., "I10").
    {"action_type": "QUERY_EHR", "module": "module_name", "query": "keyword"} - Query one hidden EHR module when the observation exposes ehr_map.
    {"action_type": "SUBMIT", "query": "code"} - Submit the final ICD-10 code (e.g., "I10").
    
    CRITICAL RULES:
    1. If your SEARCH returns empty results, try a DIFFERENT, shorter keyword. Do NOT repeat the same search.
    2. If the observation shows closed EHR modules and no decisive evidence is visible, use QUERY_EHR before guessing.
    3. Once you see the correct code in the Search Results, you MUST use the SUBMIT action.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def load_task_ids() -> list[str]:
    explicit = os.getenv("RVEDA_TASK_IDS")
    if explicit:
        task_ids = [item.strip() for item in explicit.split(",") if item.strip()]
        if task_ids:
            return task_ids

    single = os.getenv("RVEDA_TASK")
    if single:
        return [single]

    tasks_path = Path(__file__).resolve().parent / "tasks.json"
    if tasks_path.exists():
        try:
            payload = json.loads(tasks_path.read_text(encoding="utf-8"))
            task_ids = [
                item.get("task_id", "")
                for item in payload
                if isinstance(item, dict) and item.get("task_id")
            ]
            if task_ids:
                if "v2_easy_overweight_schema_v1" not in task_ids:
                    task_ids.insert(0, "v2_easy_overweight_schema_v1")
                return task_ids
        except (ValueError, OSError):
            pass

    return ["v2_easy_overweight_schema_v1", "easy_endo_1", "medium_endo_1", "hard_cardio_1"]


def normalize_score(raw_score: float) -> float:
    clamped = min(max(raw_score, 0.0), 1.0)
    if clamped <= 0.0:
        return OPEN_SCORE_MIN
    if clamped >= 1.0:
        return OPEN_SCORE_MAX
    return clamped


def parse_action_or_fallback(message: str) -> dict[str, Any]:
    fallback: dict[str, Any] = {"action_type": "SEARCH", "query": "diabetes", "module": None}

    try:
        parsed = json.loads(message)
    except json.JSONDecodeError:
        return fallback

    if not isinstance(parsed, dict):
        return fallback

    action_type = parsed.get("action_type")
    query = parsed.get("query")
    module = parsed.get("module")
    if not isinstance(action_type, str) or not isinstance(query, str):
        return fallback

    normalized_action = action_type.strip().upper()
    normalized_query = query.strip()
    if not normalized_query or normalized_action not in VALID_ACTION_TYPES:
        return fallback

    normalized_module = module.strip() if isinstance(module, str) and module.strip() else None
    if normalized_action == "QUERY_EHR" and normalized_module is None:
        return fallback

    return {
        "action_type": normalized_action,
        "query": normalized_query,
        "module": normalized_module,
    }


def build_user_prompt(step: int, history: List[str], obs) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    
    patient_note = obs.patient_note if obs else "N/A"
    search_results = obs.search_results if obs else []
    detailed_info = obs.detailed_info if obs else ""
    ehr_map = getattr(obs, "ehr_map", {}) if obs else {}
    revealed_evidence = getattr(obs, "revealed_evidence", []) if obs else []
    last_error = getattr(obs, "last_error", None) if obs else None
    invalid_reason = getattr(obs, "invalid_reason", None) if obs else None
    
    return textwrap.dedent(
        f"""
        Step: {step}
        Patient Note: {patient_note}
        Search Results: {search_results}
        EHR Map: {ehr_map}
        Revealed Evidence: {revealed_evidence}
        Detailed Info: {detailed_info}
        Last Error: {last_error}
        Invalid Reason: {invalid_reason}
        
        Previous steps history:
        {history_block}
        
        Analyze the patient note and your search results. Output your next action in strict JSON format.
        """
    ).strip()


def get_model_message(client: OpenAI, step: int, history: List[str], obs) -> str:
    user_prompt = build_user_prompt(step, history, obs)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Strip markdown formatting just in case the LLM tries to be helpful
        text = text.replace("```json", "").replace("```", "").strip()
        return text if text else '{"action_type": "SEARCH", "query": "hypertension"}'
    except Exception:
        return '{"action_type": "SEARCH", "query": "hypertension"}'


async def run_task_episode(client: OpenAI, env: RvedaEnv, task_id: str) -> None:
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = OPEN_SCORE_MIN
    success = False
    terminated_without_submit = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_id=task_id)  # OpenENV.reset()
        obs = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            message = get_model_message(client, step, history, obs)
            parsed = parse_action_or_fallback(message)

            result = await env.step(
                RvedaAction(
                    action_type=parsed["action_type"],
                    query=parsed["query"],
                    module=parsed["module"],
                )
            )
            obs = result.observation

            reward = float(result.reward) if isinstance(result.reward, (int, float)) else 0.0
            reward = min(max(reward, 0.0), 1.0)
            done = bool(result.done)
            error = getattr(obs, "last_error", None)
            final_action_type = getattr(getattr(obs, "grading", None), "action_type", "")

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=(
                    f"{parsed['action_type']}('{parsed['module']}', '{parsed['query']}')"
                    if parsed["action_type"] == "QUERY_EHR"
                    else f"{parsed['action_type']}('{parsed['query']}')"
                ),
                reward=reward,
                done=done,
                error=error,
            )

            history.append(
                (
                    f"Step {step}: QUERY_EHR(module={parsed['module']!r}, query={parsed['query']!r}) -> reward {reward:+.2f}"
                    if parsed["action_type"] == "QUERY_EHR"
                    else f"Step {step}: {parsed['action_type']}({parsed['query']!r}) -> reward {reward:+.2f}"
                )
            )

            if done:
                terminated_without_submit = final_action_type != "SUBMIT"
                break

        if not result.done:
            terminated_without_submit = True

        if terminated_without_submit:
            score = OPEN_SCORE_MIN
            success = False
        else:
            raw_score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
            score = normalize_score(raw_score)
            success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception:
        success = False
        score = OPEN_SCORE_MIN
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = await RvedaEnv.from_docker_image(IMAGE_NAME)
    task_ids = load_task_ids()

    try:
        for task_id in task_ids:
            await run_task_episode(client, env, task_id)
    finally:
        try:
            await env.close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
