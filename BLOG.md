# Rveda V2: Training a Cautious Medical Coding Agent Under Fog-of-War

Rveda V2 is a benchmark environment for **agentic medical coding under partial observability**. Instead of asking a model to emit an ICD code in one shot, the environment forces it to behave more like a cautious revenue-cycle operator:

1. reveal hidden evidence from the EHR,
2. search the ICD candidate bank,
3. inspect code details,
4. check policy and schema requirements,
5. record grounded reasoning,
6. validate the draft claim shape,
7. submit the final diagnosis code.

That workflow is the core of the benchmark. The goal is not to imitate a production medical coding platform. The goal is to make the reasoning process itself observable, auditable, and trainable.

## Why this problem matters

Medical coding is not just a classification task. It sits inside a larger operational surface where weak coding behavior can create financial, compliance, and trust problems at scale. A benchmark that rewards only the final label can easily reward the wrong policy: unsupported specificity, shortcut retrieval, and submission without enough evidence.

Rveda is designed to test the opposite behavior:

- retrieve before committing,
- cite evidence before submitting,
- adapt when the schema or policy changes,
- expose structured reward signals instead of a single opaque score.

That is why the environment includes **Fog-of-War**, **policy/schema drift**, **reasoning logs**, and **claim-schema validation** instead of only `SEARCH` and `SUBMIT`.

## What the environment does

Rveda uses OpenEnv and exposes a tool-style action loop. Depending on the task slice, the agent can:

- `QUERY_EHR`
- `SEARCH`
- `DETAILS`
- `CHECK_POLICY`
- `VALIDATE_CLAIM_SCHEMA`
- `REASONING_LOG`
- `SUBMIT`

The important design choice is that the correct answer is intentionally **not fully visible at reset**. The agent has to reveal evidence, inspect candidates, and satisfy schema constraints before it can terminate cleanly. That makes the benchmark more useful for RL than a static label lookup task.

## What we trained

The first working training proof intentionally uses a **small model** and a **small number of GRPO steps**. That was a deliberate choice:

- small models iterate faster,
- Colab reruns are more reliable,
- reward/debug loops matter more than headline model size in the early phase,
- a rerunnable proof is more valuable than one oversized run that barely fits.

The current training path uses:

- OpenEnv for the environment layer,
- a Colab-rerunnable notebook launcher,
- TRL-based GRPO,
- QLoRA-style low-memory loading,
- generated V2 tasks as a controlled curriculum source.

## What worked

The strongest current Colab smoke run completed end to end on a **Tesla T4** using `Qwen/Qwen2.5-1.5B-Instruct` and produced:

- `summary.json`
- `baseline_model_eval.json`
- `post_train_model_eval.json`
- `trainer_log_history.json`
- `baseline_vs_trained_comparison.json`
- reward/loss/verifier plots

That run evaluated **4 generated tasks**, reached `SUBMIT` on all `4`, achieved a **search-to-submission ratio of 1.0**, and finished with **0 timeouts**. This is the important milestone: the environment, the trainer, and the saved artifacts are all real and rerunnable.

## What the current numbers do and do not show

The most honest summary of the current best smoke run is:

- baseline mean total reward: **1.32375**
- trained mean total reward: **1.31500**
- delta: **-0.00875**

So the current result proves that:

- the OpenEnv environment is live,
- the trainer is live,
- the generated tasks are usable for training,
- the saved artifacts are real and reviewable,
- the policy can complete full trajectories through `SUBMIT`.

But it does **not** prove that the learned 1.5B policy is already stronger than the scripted baseline. In the current smoke configuration, the scripted baseline remains slightly better.

That distinction matters. The current result should be read as a **reproducible training-proof milestone**, not as a final performance win.

## What still needs to improve

The most important current gap is not model size. It is **turning a stable training proof into an actual learning win**.

Now that the generated-task run reaches `SUBMIT`, the next iteration should focus on:

- improving reward shaping so the learned policy can beat the scripted baseline,
- expanding the curriculum while keeping the Colab path rerunnable,
- making schema-validation and drift metrics show up more clearly in the results,
- only then scaling to larger presets.

That is a much better use of compute than immediately jumping to 7B or 14B.

## Why OpenEnv matters here

One of the non-negotiables for this project is using the latest OpenEnv release and building on top of the framework rather than inventing a custom environment stack. Rveda does that. The environment is packaged as an OpenEnv-compatible server and validated through the OpenEnv tooling, which makes it easier for reviewers to inspect, rerun, and compare against other submissions.

## Reviewer-facing assets

The core submission materials are:

- the Hugging Face Space,
- the README,
- the Colab training notebook,
- the saved plots and summary artifacts,
- this mini-blog or a published version of it,
- an optional short video showing the environment loop and training results.

The repo should stay lightweight. Large videos should not be committed here; link to public URLs instead.

## The short version

Rveda V2 is a benchmark for training **careful, evidence-seeking medical coding agents** rather than one-shot label emitters. The environment already supports the critical interaction loop, the Colab training path reruns on a small model, and the strongest current smoke run proves real end-to-end training and submission behavior. The remaining challenge is not getting the pipeline to run; it is getting the learned policy to outperform the scripted baseline.
