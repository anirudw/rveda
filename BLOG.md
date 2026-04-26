# RVEDA RCM ARENA: Training a Medical Coding Agent Under Hidden Evidence and Policy Drift

## Why Medical Coding Needs an Interactive Benchmark

Medical coding can look deceptively simple if you reduce it to label prediction: read a note, emit an ICD code, score exact match. Real revenue-cycle work is harder. Relevant evidence is often scattered across chart sections, payer requirements can change, near-neighbor codes can look plausible, and unsupported specificity can create financial and compliance problems rather than just a benchmark miss.

RVEDA RCM ARENA models that operational loop instead of the static label problem. That is why it fits OpenEnv and RL well. The task is interactive by nature: the agent has to reveal evidence, search a large code space, inspect details, adapt to drift, and submit a grounded claim under a structured reward function. The question is not only whether the model can name a code. It is whether it can behave like a cautious coder when the easy shortcut is to guess.

There is real business value behind that framing. In a [JAMA study of billing and insurance-related work](https://jamanetwork.com/journals/jama/fullarticle/2673148), administrative processing was estimated at `$20` to `$215` per encounter and `3%` to `25%` of professional revenue depending on encounter type. In Medicare Advantage, [MedPAC's March 2024 report](https://www.medpac.gov/document/chapter-13-estimating-medicare-advantage-coding-intensity-and-favorable-selection-march-2024-report/) estimated a projected `$83 billion` payment gap versus FFS in 2024, with coding intensity alone contributing about `$50 billion`. A [2025 Health Affairs Scholar study](https://pubmed.ncbi.nlm.nih.gov/39822237/) found wide variation in coding inflation across contracts. RVEDA RCM ARENA is not a production claims-control system, but those figures make the benchmark target worth taking seriously.

## How RVEDA RCM ARENA Turns Coding Into a Tool-Use Problem

RVEDA RCM ARENA is an OpenEnv environment for agentic medical coding under partial observability. The episode does not start with a fully exposed chart and a clean target label. The agent has to work toward a defensible submission.

The loop is simple to describe and hard to execute well. `QUERY_EHR` reveals hidden chart evidence. `SEARCH` and `DETAILS` navigate the ICD candidate space and exclusions. `CHECK_POLICY` and claim validation expose schema requirements that can change mid-episode. The trajectory only ends cleanly when the agent records grounded reasoning, validates the draft claim shape, and `SUBMIT`s the final code.

## Why This Is Harder Than Label Prediction

RVEDA RCM ARENA is not difficult because the label set is large. It is difficult because it punishes the shortcuts that flatter benchmarks often reward.

If evidence is hidden, guessing early can land on a plausible but unsupported code. If the schema changes mid-episode, a previously acceptable draft can become invalid. If the model submits a near-neighbor code from the right family without the right evidence, that should still count as failure in operational terms. The verifier therefore cares about more than terminal text: it grades correctness, grounding, and whether the workflow itself was valid.

The model is not just selecting a label from text. It is making sequential decisions under uncertainty while trying to avoid incorrect but superficially reasonable actions.

## Training Configuration

The current training path is deliberately small-model first. The strongest rerunnable evidence comes from `Qwen/Qwen2.5-1.5B-Instruct` on a Tesla T4 in Colab, trained with TRL GRPO over live environment rollouts using [train_grpo_smoke.py](train_grpo_smoke.py) and the generated-task notebook linked above.

We started with an Unsloth-assisted path because it is attractive for memory-constrained RL, especially on Colab hardware. In practice, the most stable setup for this project became a plain TRL fallback with low-memory loading. That was an engineering tradeoff, not a branding choice. At this phase, reproducibility mattered more than saying we ran the largest stack possible.

The reward path is live through the environment bridge rather than an offline label file. RVEDA RCM ARENA is meant to grade behavior, not just final text, so the trainer needs to interact with the environment reward surface.

Current scale is still early. The strongest run so far used 4 generated tasks, 8 train steps, and an 8-episode smoke evaluation. That is enough to show a meaningful behavior change, but not enough to treat the result as a final performance claim.

## First Failure Modes

The first real failure was pipeline quality, not model quality. The generated-task Colab notebook did not work from a fresh runtime because it tried to install from `/content/rveda` before cloning the repo. In a judged environment, that kind of rerun brittleness matters.

The next failure was more instructive. The Unsloth FastRL path loaded and then crashed inside GRPO with a compatibility mismatch around missing `old_logps` and `ref_logps`. Passing `--disable-unsloth-fast-rl` was not enough because the initial fallback still flowed through parts of the Unsloth stack. We had to build a real plain-TRL fallback.

That fallback then hit repeated `Float` versus `BFloat16` errors on the T4 during generation and training. A small-model Colab RL path is not automatically stable just because the model fits in memory.

Even after the trainer ran, an early tiny run produced misleading optics. It showed a small reward increase, from `0.18075` to `0.2`, but the agent never reached `SUBMIT`. The reason was simple: the episode budget was only 4 steps, which made the full cautious workflow impossible by construction. The run was not fake, but it proved less than the raw reward delta suggested.

There were quieter failures too. Some tests were brittle because they depended on task ordering rather than explicit task lookup. Several judge-friendly metrics remained proxy-only: the smoke runner emitted a grounding proxy, while drift adaptation and schema validation pass rates remained null in the strongest smoke runs.

## Lessons from the Failures

The main lesson was that training credibility depends as much on observability as on reward design.

The low-step run showed that a benchmark can produce "training evidence" while making terminal success impossible. That changed the curriculum. Step budgets, action requirements, and terminal workflow validity have to be designed together.

The trainer failures taught us not to confuse a nominal stack choice with a stable one. "Plain TRL fallback on a 1.5B model" turned out to be more valuable than a more impressive but brittle setup because it reran, produced artifacts, and exposed real behavior.

The metric gaps led to a similarly practical conclusion. A grounding proxy is not a real grounding F1. Null drift adaptation and schema validation rates do not mean those dimensions are unimportant; they mean the runner still needs stronger labels and richer event traces. That changed the work plan. The next improvement is not just more training. It is better metric instrumentation.

## Current Evidence

What we can honestly show today is concrete:

- a runnable Hugging Face Space
- a rerunnable Colab notebook
- a working training script with a plain-TRL fallback path
- saved artifacts including `summary.json`, baseline and post-train eval outputs, trainer logs, and comparison files
- reward, loss, and verifier plots from a real Colab run

The strongest smoke run used `Qwen/Qwen2.5-1.5B-Instruct` on 4 generated tasks with 8 train steps and an 8-episode evaluation. Baseline mean total reward was `0.810125`; post-train mean total reward was `1.31500`, a delta of `+0.504875`. The trained policy reached `SUBMIT` on `8 / 8` episodes, while the scripted baseline only completed `4 / 8` and timed out on half of them. Search-to-submission improved from `3.25` to `1.0`, and timeout frequency dropped from `0.5` to `0.0`.

That is materially stronger evidence than the earlier runs. At the same time, it is still smoke-scale evidence. Grounding is still tracked through a proxy, and that proxy dipped slightly from `0.8889` to `0.8824`, so the strongest claim we can make is that the learned policy became much more reliable at completing the workflow, not that every quality dimension improved at once.

## What Is Real Today

Several things are no longer hypothetical. The environment runs on OpenEnv, exposes the intended action loop, and supports hidden evidence, code retrieval, policy and schema checks, and grounded submission. A judge can rerun the notebook on commodity Colab hardware and reproduce a working smoke training cycle. The run produces saved comparisons and plots rather than relying on an ephemeral output cell.

Just as importantly, the environment already exposes failure modes that a flatter benchmark would hide: exploration without submission, submission without enough evidence, and metric surfaces that are still too weakly labeled to support stronger claims. The latest run also shows that training can improve completion behavior in a measurable way.

## Still Early

The project is still early in three ways. First, the generated curriculum is still small. It is enough for smoke training and debugging, not enough for a confident learning claim. Second, several of the most interesting metrics still need stronger labels and richer event capture, especially for grounding, drift adaptation, and schema validation. Third, we have intentionally not centered the story around 7B or 14B training because the present bottleneck is not raw parameter count but reliable iteration and stronger evaluation labels.

Larger hardware and larger models may matter later, especially for broader generated curricula and more complex drift behavior. But scaling before the metric surface and rerun path are stable would be the wrong optimization.

## Why the Environment Still Matters

RVEDA RCM ARENA is promising because it is asking the right question. Hidden evidence, near-neighbor codes, and claim-format drift create a more meaningful evaluator than a static lookup benchmark. The verifier structure already gives a clear path from "the pipeline runs" to "the learned policy improves for the right reasons."

That makes the current work useful even before it becomes strong on headline reward. The project already surfaces which parts of cautious medical coding are easy to fake, which parts need richer supervision, and which metrics should be treated as provisional until the environment emits stronger labels.

## Where RVEDA RCM ARENA Stands Now

RVEDA RCM ARENA today is a real OpenEnv environment with real training infrastructure, not a speculative benchmark sketch. It already supports rerunnable interaction, saved training artifacts, and end-to-end submission behavior on generated tasks. The latest smoke run also shows a real policy improvement on completion-oriented metrics over the scripted baseline.

That is exactly why the current stage matters. The environment is stable enough to reveal failure modes, honest enough to show mixed metric movement rather than only headline gains, and structured enough to scale into stronger runs once the data, metrics, and curriculum improve. For OpenEnv Round 2, that is the core claim: not that the problem is solved, but that the right environment has been built, trained, and learned from.
