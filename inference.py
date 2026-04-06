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
from typing import List, Optional

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
TASK_NAME = os.getenv("RVEDA_TASK", "medical_coding")
BENCHMARK = os.getenv("RVEDA_BENCHMARK", "rveda")
MAX_STEPS = 8
TEMPERATURE = 0.7
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.1  # normalized score in [0, 1]
MAX_TOTAL_REWARD = 1.0

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert Medical Coder agent.
    Your task is to review a patient note, search the ICD-10 taxonomy, and SUBMIT the most accurate code.
    
    You must reply ONLY with a valid JSON object matching this schema:
    {"action_type": "SEARCH", "query": "keyword"} - Use short, single-word keywords (e.g., "hypertension").
    {"action_type": "DETAILS", "query": "code"} - Look up rules for a specific code (e.g., "I10").
    {"action_type": "SUBMIT", "query": "code"} - Submit the final ICD-10 code (e.g., "I10").
    
    CRITICAL RULES:
    1. If your SEARCH returns empty results, try a DIFFERENT, shorter keyword. Do NOT repeat the same search.
    2. Once you see the correct code in the Search Results, you MUST use the SUBMIT action.
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


def build_user_prompt(step: int, history: List[str], obs) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    
    patient_note = obs.patient_note if obs else "N/A"
    search_results = obs.search_results if obs else []
    detailed_info = obs.detailed_info if obs else ""
    
    return textwrap.dedent(
        f"""
        Step: {step}
        Patient Note: {patient_note}
        Search Results: {search_results}
        Detailed Info: {detailed_info}
        
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
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return '{"action_type": "SEARCH", "query": "hypertension"}'


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = await RvedaEnv.from_docker_image(IMAGE_NAME)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset() # OpenENV.reset()
        obs = result.observation   # Initialize obs BEFORE the loop!
        last_reward = 0.0          # Initialize last_reward BEFORE the loop!

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            # Pass only the necessary arguments matching our new function signature
            message = get_model_message(client, step, history, obs)

            try:
                parsed = json.loads(message)
            except json.JSONDecodeError:
                # Fallback to prevent crash if LLM hallucinates
                parsed = {"action_type": "SEARCH", "query": "hypertension"}

            result = await env.step(
                RvedaAction(action_type=parsed["action_type"], query=parsed["query"])
            )
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(
                step=step,
                action=f"{parsed['action_type']}('{parsed['query']}')",
                reward=reward,
                done=done,
                error=error,
            )

            history.append(
                f"Step {step}: {parsed['action_type']}({parsed['query']!r}) -> reward {reward:+.2f}"
            )

            if done:
                break

        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)  # clamp to [0, 1]
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())