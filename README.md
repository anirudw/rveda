---
title: Rveda Environment Server
emoji: "🏥"
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - medical-coding
  - agentic-auditing
---

# Rveda

**Rveda is a benchmark environment for agentic medical coding in a human-in-the-loop setting.** It evaluates whether an LLM agent can read a clinical note, retrieve the right evidence from a local ICD-10 knowledge source, and submit the most accurate code with a traceable decision path.

The repository is intentionally lightweight: the current build uses a local SQLite-backed ICD-10 database seeded with mock data, an OpenEnv-compatible environment server, and a reference inference loop built on the OpenAI client. That makes Rveda well suited for benchmarking clinical reasoning, retrieval efficiency, and tool-use discipline under deterministic conditions.

Although the included baseline is a single-agent loop, the environment is structured for broader multi-agent experimentation, such as retriever-coder-auditor pipelines, while preserving a standardized evaluation interface.

## Deployment Status

- Active Round 2 development Space: `https://huggingface.co/spaces/anirudw/rveda-rcm-arena`
- Archived Round 1 baseline Space: `https://huggingface.co/spaces/anirudw/rveda`
- This repository currently contains the active OpenEnv migration work and the frozen V2 contract artifacts, but the live runtime is still mostly the Round 1 medical-coding environment while Fog-of-War, policy drift, and structured claim mechanics are implemented.
## Round 2 Status

Rveda V2 is planned as **Revenue Cycle Drift Arena**, a Round 2 professional-task environment for partially observable revenue-cycle workflows.

- Round 2 Space target: [`anirudw/rveda-rcm-arena`](https://huggingface.co/spaces/anirudw/rveda-rcm-arena)
- Round 1 archived baseline: [`anirudw/rveda`](https://huggingface.co/spaces/anirudw/rveda)
- V2 contract: [`docs/rveda-v2-contract.md`](docs/rveda-v2-contract.md)
- Minimal V2 task example: [`examples/v2_task_minimal.json`](examples/v2_task_minimal.json)

The default runtime remains the Round 1 medical-coding environment. The repository now includes a minimal V2 slice behind the explicit task ID `v2_easy_overweight_schema_v1`: hidden EHR modules, `QUERY_EHR`, `ehr_map`, revealed evidence, `CHECK_POLICY`, `VALIDATE_CLAIM_SCHEMA`, visible `policy_state`, schema-drift notices, `REASONING_LOG`, and grounding-gated submit behavior are implemented. Later V2 mechanics such as structured claim submission remain unimplemented.

## Training Path

The judge-facing training path is intentionally small-model first so it can be rerun in Colab instead of depending on one large-model attempt.

- Generated V2 Colab launcher: [`train_generated_v2_grpo_launcher.ipynb`](train_generated_v2_grpo_launcher.ipynb)
- Smoke Colab launcher: [`train_grpo_smoke_launcher.ipynb`](train_grpo_smoke_launcher.ipynb)
- Training runner: [`train_grpo_smoke.py`](train_grpo_smoke.py)

Recommended order:

1. Sync the repo in Colab.
2. Run `python -m pytest -q`.
3. Run `openenv validate`.
4. Start with the tiny `Qwen/Qwen2.5-1.5B-Instruct` preset and plain TRL fallback.
5. Confirm that the run produces `summary.json`, `baseline_model_eval.json`, `post_train_model_eval.json`, `loss_plot.svg`, `reward_plot.svg`, and `verifier_metrics_plot.svg`.
6. Scale to larger presets only after the small-model run is stable.

The README already links the active Hugging Face Space. When the external mini-blog, short video, or slide deck is ready, add those public URLs here rather than committing large media files into the repo.

## Why Rveda

Medical coding sits inside a much larger operational and financial surface area. A UC San Diego and *Health Affairs* analysis projected that aggressive diagnostic coding intensity could drive [more than $200 billion in Medicare overpayments over a decade](https://www.sciencedaily.com/releases/2017/02/170207092727.htm). A recent Zinnov industry report similarly projects U.S. healthcare revenue cycle management spend to reach [USD 200-210 billion by 2029](https://zinnov.com/centers-of-excellence/the-200-billion-question-why-the-future-of-healthcare-rcm-may-belong-to-india) as billing workflows become more fragmented and administratively heavy. Those figures do not imply that a lightweight benchmark solves the full problem, but they do show why coding behavior is not a harmless toy task: inaccurate, over-aggressive, or weakly verified coding decisions can scale into real financial and operational damage.

A benchmark that rewards only the final label risks training exactly the wrong behavior: hallucinating or overly aggressive agents that maximize apparent specificity without grounding. Rveda is designed to test the opposite behavior: grounded, stepwise coding decisions in which retrieval and verification are part of the task, not optional post-processing.

Rveda is designed to answer a concrete research question:

> Can an LLM agent behave like a cautious medical coder, rather than a one-shot label generator?

This framing matters for benchmark design:

- It tests **clinical reasoning**, not only label recall.
- It tests **search efficiency**, because the agent must retrieve and inspect evidence before submission.
- It penalizes **hallucinated or over-aggressive coding behavior** by making verification part of the interaction loop.
- It supports **human-in-the-loop auditing**, because each step leaves an explicit interaction trace.

## Market Context: Auditing vs. Benchmarking

Established platforms such as FraudLens, Cotiviti, and Optum FWA address a different layer of the problem: post hoc detection of fraud, waste, abuse, and anomalous billing behavior across large claims datasets.

Rveda addresses a different question. It is a **pre-deployment benchmark** for agentic medical coding systems, designed to test whether an AI model arrives at a code through grounded clinical reasoning before deployment.

That distinction is important. Statistical anomaly detection evaluates aggregate billing behavior across populations and claims streams; Rveda evaluates the reasoning trajectory of an individual AI agent as it searches, inspects evidence, and commits to an ICD-10 code. In that sense, Rveda is complementary to enterprise auditing systems: those platforms help catch problematic claims after the fact, while Rveda is designed to test whether an autonomous coding agent should be trusted before deployment.

## Benchmark Task

Each episode starts with a patient note and ends when the agent submits an ICD-10 code or exhausts the episode budget.

The action space is deliberately small and tool-like:

- `SEARCH(query)`: query the local ICD-10 index for candidate codes.
- `DETAILS(code)`: retrieve long-form code details and exclusion notes.
- `QUERY_EHR(module, query)`: reveal evidence from one hidden EHR module in the minimal Task 1.3 V2 slice.
- `CHECK_POLICY()`: reveal the active payer policy version and current claim-schema requirements.
- `VALIDATE_CLAIM_SCHEMA(payload)`: validate a draft claim against the active schema without ending the episode.
- `REASONING_LOG(payload)`: submit a grounded reasoning record that cites revealed evidence before final submission.
- `SUBMIT(code)`: finalize the coding decision and end the episode.

This setup mimics the operational logic of medical coding review: reveal hidden evidence when needed, retrieve candidates, inspect details, check the active claim rules, validate a draft claim, record grounded reasoning, then commit.

## Architecture

Rveda consists of three core layers: a local retrieval engine, an environment wrapper with grading logic, and a reference inference loop.

### 1. Local ICD-10 Engine: `server/engine.py`

`server/engine.py` is the retrieval backend used by the environment.

- `initialize_db()` creates `data/icd10.db` and seeds it from `icd10_mock.json`.
- The SQLite table stores `code`, `short_desc`, `long_desc`, and `excludes`.
- `search_codes(query, limit=5)` performs lexical retrieval over `short_desc` and `long_desc` using SQLite `LIKE` matching and returns compact candidate summaries.
- `get_code_details(code)` performs exact-code lookup and returns long description plus exclusion notes.

This design is intentionally simple and reproducible. The database is local, deterministic, and fast enough for benchmark-grade evaluation without introducing external search infrastructure.

### 2. Environment and Reward Logic: `server/rveda_environment.py`

`server/rveda_environment.py` wraps the engine in an OpenEnv-compatible task environment.

- On startup, it calls `initialize_db()` so the packaged SQLite database is ready before episodes begin.
- `reset()` loads a task from `tasks.json` and exposes the patient note as the initial observation.
- `step()` routes each action to the proper backend operation and returns structured observations containing search results, detailed code context, reward, and grading metadata.

The environment also records a rich `GradingTrace`, including:

- difficulty tier,
- search history,
- code inspection history,
- most recent search candidates,
- reward components,
- conflict flags such as `Excludes1` mismatches.

This makes Rveda useful not only for final-score benchmarking, but also for trajectory-level analysis of how an agent reasoned through the task.

### 3. Reference Inference Loop: `inference.py`

`inference.py` provides the benchmark submission loop.

At runtime it:

1. Reads task IDs from `tasks.json` or from `RVEDA_TASK` / `RVEDA_TASK_IDS`.
2. Creates an OpenAI-compatible client using `HF_TOKEN` (or `API_KEY`), `API_BASE_URL`, and `MODEL_NAME`.
3. Launches the environment with `RvedaEnv.from_docker_image(IMAGE_NAME)`.
4. Resets into an episode and builds a prompt from the current patient note, search results, detailed info, policy state, drift notice, reasoning-log status, and recent action history.
5. Asks the model to emit strict JSON with one of seven actions: `SEARCH`, `DETAILS`, `QUERY_EHR`, `CHECK_POLICY`, `VALIDATE_CLAIM_SCHEMA`, `REASONING_LOG`, or `SUBMIT`.
6. Executes the action in the environment, logs `[START]`, `[STEP]`, and `[END]` lines, and repeats until termination.

The loop is intentionally benchmark-friendly: it is deterministic in structure, OpenAI-client compliant, and emits normalized episode scores for consistent downstream evaluation.

## Benchmarking and Scoring

Rveda is designed around two measurable axes:

- **Accuracy**: did the agent submit the correct ICD-10 code, or at least the correct code family?
- **Efficiency**: how economically did the agent search, inspect, and commit within a bounded number of steps?

The environment also exposes rubric-level `reward_metrics` so terminal correctness, grounding, schema compliance, format validity, process discipline, and drift adaptation can be inspected independently.

### Accuracy Signal

Submission quality is the dominant grading signal.

- Exact-code submissions receive the highest base reward.
- Same-family submissions receive partial credit.
- Incorrect-family submissions receive a lower score.

This reflects a realistic coding hierarchy: selecting the right diagnostic family is better than an unrelated code, but full specificity still matters.

### Efficiency Signal

Rveda also scores process quality before submission.

- Novel, productive searches earn small bonuses.
- Relevant detail lookups earn additional reward.
- Repeated low-value exploration stops improving the score.
- Episodes are capped at **8 steps**, and failing to submit within budget ends the episode.

In benchmark terms, this acts as a **step penalty**: extra actions consume the fixed interaction budget, reduce the value of aimless search, and increase the risk of timing out before a valid `SUBMIT`. Faster, better-grounded coding trajectories therefore outperform slow or repetitive ones.

### Score Normalization

The final episode score reported by `inference.py` is normalized to a bounded 0-1 scale so tasks remain comparable across runs.

Episode rewards are standardized before reporting, which keeps evaluation stable while preserving relative ranking between stronger and weaker coding trajectories.

### Why the Scoring Design Matters

The benchmark therefore rewards:

- correct final coding decisions,
- efficient evidence gathering,
- auditable trajectories,
- compliance with strict evaluation contracts.

## Task Specification

Tasks are defined in `tasks.json` as JSON objects with four fields:

- `task_id`
- `difficulty`
- `patient_note`
- `target_code`

The current benchmark ships with a simple 3-tier structure:

| Tier | Example Task | Clinical Pattern | Target Code |
| --- | --- | --- | --- |
| Easy | `easy_endo_1` | Routine visit with elevated BMI and weight-management counseling | `E66.3` |
| Medium | `medium_endo_1` | Autoimmune hypothyroid presentation consistent with Hashimoto's disease | `E06.3` |
| Hard | `hard_cardio_1` | Acute myocardial infarction presentation in the emergency setting | `I21.9` |

This tiered task structure is useful for benchmarking both capability and scaling behavior: simple lexical retrieval may be enough on easy cases, while harder tasks require better grounding and more disciplined tool use.

## Setup

### Prerequisites

- Python 3.10+
- Docker
- `uv` or a compatible Python package installer

### 1. Install dependencies

```bash
uv sync
```

### 2. Initialize the local SQLite database

The environment initializes the database automatically at startup, but you can also prebuild it explicitly:

```bash
python -c "from server.engine import initialize_db; initialize_db()"
```

This creates `data/icd10.db` from the mock ICD-10 records in `icd10_mock.json`.

### 3. Build the environment image

```bash
docker build -t rveda-env:latest -f Dockerfile .
```

### 4. Optional: validate the environment

```bash
openenv validate
```

## Usage

### Run the server locally

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Run the reference inference loop

Set the required environment variables first.

#### Bash

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="<your_hf_token>"
export IMAGE_NAME="rveda-env:latest"
python inference.py
```

#### PowerShell

```powershell
$env:API_BASE_URL = "https://router.huggingface.co/v1"
$env:MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
$env:HF_TOKEN = "<your_hf_token>"
$env:IMAGE_NAME = "rveda-env:latest"
python inference.py
```

Optional task controls:

- `RVEDA_TASK=<task_id>` runs a single task.
- `RVEDA_TASK_IDS=<task_a,task_b,...>` runs a selected task set.

During execution, `inference.py` prints benchmark-compatible logs in the form:

```text
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP] step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>
```

## Repository Layout

- `server/engine.py`: SQLite-backed ICD-10 retrieval and detail lookup
- `server/rveda_environment.py`: environment state machine and reward shaping
- `server/app.py`: FastAPI / OpenEnv server wrapper
- `client.py`: client interface for interacting with the environment
- `models.py`: action, observation, and grading schemas
- `inference.py`: OpenAI-client baseline loop
- `tasks.json`: benchmark task suite
- `icd10_mock.json`: mock ICD-10 source data
- `data/icd10.db`: generated SQLite database used at runtime
- `docs/rveda-v2-contract.md`: frozen V2 Task 0.1 schema and verifiability contract
- `examples/v2_task_minimal.json`: minimal V2 curriculum slice example

## Scope and Limitations

Rveda is a **benchmarking environment**, not a production clinical coding system.

- The current ICD-10 corpus is mock data.
- Retrieval is lexical and SQLite-backed rather than semantic or ontology-scale.
- The SQLite backend is an intentional benchmark constraint: it keeps the environment local, deterministic, lightweight, and reproducible while forcing agents to reason under limited search conditions.
- The included agent loop is a baseline, not a claim of clinical deployment readiness.

Those constraints are a feature, not a flaw: they keep the benchmark controlled, portable, and easy to reproduce while still exercising the core reasoning loop of agentic medical coding.

## Research Use Cases

Rveda is well suited for:

- benchmarking LLM agents on coding accuracy under constrained search,
- comparing single-agent and multi-agent coding strategies,
- studying tool-use efficiency under a fixed step budget,
- auditing reasoning traces in human-in-the-loop evaluation,
- testing stable scoring pipelines for controlled benchmark environments.
