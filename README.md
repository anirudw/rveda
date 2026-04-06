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

# Rveda: The Medical Coding Flight Simulator

Training Agentic Auditors to Solve the $210B Healthcare Administrative Crisis.

## The Problem Statement

Medical coding is one of the highest-friction bottlenecks in healthcare operations. Clinical notes must be translated into precise billing codes, but the process is expensive, slow, and error-prone. Industry estimates regularly point to billing error rates that affect a large majority of claims workflows, with roughly 80% of bills containing some form of error, omission, mismatch, or preventable rework. The financial consequence is enormous: healthcare administrative waste is estimated in the hundreds of billions of dollars annually, with this submission framing the coding and claim-quality problem as part of a $210B yearly burden.

Human coders remain essential, but throughput is limited. In realistic review settings, a trained professional may only process around 18 charts per day when accuracy, compliance, and auditability are required. That makes the problem a poor fit for pure manual scaling and a dangerous fit for naive automation.

Rveda is designed as a training environment for agents that must reason through coding decisions rather than guessing them.

## The Core Philosophy

Rveda is built around a simple thesis: medical coding should be treated as an auditing problem, not a one-shot text classification problem.

| Dimension | Static Classification | Agentic Auditing |
| --- | --- | --- |
| Representative approach | Single-pass label prediction, similar to systems like FraudLens-style classifiers | Multi-step evidence gathering and verification |
| Core behavior | Predict a code directly from the note | Search for candidate codes, inspect details, then submit |
| Failure mode | Confident wrong answer from pattern matching | Slower, but auditable and evidence-backed decisions |
| Hallucination risk | High when the model skips verification | Lower because the agent is forced through `SEARCH` and `DETAILS` |
| Training signal | Often sparse and end-state only | Dense and process-aware |
| Interpretability | Weak | Stronger step-by-step trail |
| Rveda position | Not the target paradigm | The target paradigm |

Rveda forces the agent to behave like an auditor. The environment exposes explicit tool-like actions, making it costly to hallucinate and beneficial to verify. Instead of rewarding eloquent guesses, it rewards grounded interaction with retrieval evidence before final submission.

## Environment Design

Rveda is an OpenEnv environment for episodic medical coding tasks. Each episode starts from a patient note and ends when the agent submits a code or reaches the step limit.

### Action Space

The environment exposes three actions:

- `SEARCH(query)`: Retrieve candidate ICD-10 codes using the local search engine.
- `DETAILS(code)`: Inspect the long-form description and exclusion notes for an exact code.
- `SUBMIT(code)`: Commit to a final ICD-10 code and terminate the episode.

This action design intentionally separates retrieval, inspection, and commitment. The environment is not asking the agent to emit a final label immediately; it is asking the agent to act like a verification-first coder.

### Observation Space

The observation schema is structured around the progressive disclosure of evidence:

- `patient_note`: The raw clinical scenario visible at reset.
- `search_results`: Candidate code matches surfaced by `SEARCH`.
- `detailed_info`: Code-specific explanatory context surfaced by `DETAILS`.
- `current_reward`: The immediate reward attached to the current observation.

Operationally, the interaction flow is:

`patient_note -> search_results -> detailed_info -> submit`

That flow is the core learning curriculum. Agents must learn when to broaden search, when to drill into a code, and when there is enough evidence to submit.

## Hierarchical Reward Logic

Rveda uses a dense hierarchical reward rather than a purely binary success signal.

- `1.0`: Exact code match with the episode target.
- `0.5`: Partial category match when the first three characters align with the target family.
- `0.0`: Incorrect submission outside the target category.

### Partial Progress Signal

The partial reward is deliberate. In real coding workflows, identifying the correct disease family but missing final specificity is materially better than selecting an unrelated diagnosis. The reward model reflects that difference.

This is superior for RL training compared with a sparse binary signal because it:

- gives the policy gradient information before full mastery,
- distinguishes clinically adjacent errors from random failures,
- encourages structured exploration over blind guessing,
- makes it easier to train agents on difficult specificity jumps within a code family.

In short, the reward system measures progress, not just perfection.

## Technical Constraints & Future Work

This competition version is optimized for portability and tight resource budgets. The current design uses a lightweight SQLite full-text search style backend to stay compatible with an 8GB RAM competition constraint while still supporting local retrieval and deterministic evaluation.

That constraint is architectural, not conceptual. The environment has been structured so the retrieval engine is hot-swappable:

- Competition build: SQLite-backed local retrieval
- Production candidate: Vector database such as Pinecone or Milvus
- Production retrieval model: Clinical embeddings such as BioBERT or related domain-tuned encoders

This means the benchmark loop can stay stable while the retrieval substrate upgrades from lightweight lexical search to semantic clinical retrieval.

Near-term future work includes:

- richer ICD-10 coverage beyond the mock dataset,
- multi-code episodes and sequencing constraints,
- payer-specific rules and denial-aware evaluation,
- clinician-query actions when documentation is insufficient,
- stronger automated tests and benchmark reporting,
- production retrieval adapters for vector search backends.

## Reproduction

### 1. Build the environment image

From the repository root:

```bash
docker build -t rveda-env:latest -f Dockerfile .
```

You can also validate the environment locally before running inference:

```bash
openenv validate
```

### 2. Configure inference credentials

Set the required environment variables before running the agent loop:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="<your_token>"
export IMAGE_NAME="rveda-env:latest"
```

On PowerShell:

```powershell
$env:API_BASE_URL = "https://router.huggingface.co/v1"
$env:MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
$env:HF_TOKEN = "<your_token>"
$env:IMAGE_NAME = "rveda-env:latest"
```

### 3. Run inference

Execute the submission loop from the project root:

```bash
python inference.py
```

The script will:

- create the OpenAI-compatible client,
- launch the environment from the Docker image,
- reset into a coding task,
- prompt the model to emit raw JSON actions,
- execute `SEARCH`, `DETAILS`, and `SUBMIT` steps,
- print standardized `[START]`, `[STEP]`, and `[END]` lines for evaluation.

## Repository Map

- [`models.py`](/X:/Projects/rveda/models.py): Action and observation schema
- [`client.py`](/X:/Projects/rveda/client.py): OpenEnv client wrapper
- [`server/app.py`](/X:/Projects/rveda/server/app.py): HTTP server entrypoint
- [`server/rveda_environment.py`](/X:/Projects/rveda/server/rveda_environment.py): Core environment logic
- [`server/engine.py`](/X:/Projects/rveda/server/engine.py): SQLite-backed retrieval engine
- [`tasks.json`](/X:/Projects/rveda/tasks.json): Episode/task definitions
- [`icd10_mock.json`](/X:/Projects/rveda/icd10_mock.json): Mock ICD-10 code corpus
- [`inference.py`](/X:/Projects/rveda/inference.py): Competition inference loop

## Current Scope

Rveda is already a usable OpenEnv submission: it validates, runs locally, exposes a structured action interface, and supports agent training on evidence-backed coding behavior. It is not yet a full production medical coding platform, and it does not claim to be. Its current role is more precise: a flight simulator for agentic medical auditing under realistic operational constraints.
