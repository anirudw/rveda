# Rveda v2 - Task Backlog & Roadmap

**Category:** guide
**Tags:** rveda, backlog, roadmap, sprint
**Last updated:** 2026-04-25
**Related:** [Rveda Development Hub](index.md), [System Requirements](requirements.md), [Architecture Audit](../rveda-analysis/prompt-01-architecture.md)

---

## Sprint 0: Submission Guardrails
*Focus: Make the plan judge-compatible before implementation detail spreads.*

## Backlog Operating Rules
- Treat this file as the primary execution tracker for both developers. Each task has one primary owner and one review handoff.
- `Sprint 0` is blocking for round 2 compliance. Only `Task 3.1` template work may proceed in parallel, and only after `Task 0.1` freezes the v2 schema.
- Prefer demoable vertical slices over speculative optimizations. A runnable Space, a rerunnable training script, and readable evidence beat a larger unfinished design.
- Keep the Round 1 Space (`anirudw/rveda`) as the archived baseline. Round 2 must ship on a dedicated submission Space with its own stable URL.

## Two-Developer Setup
- **Developer A - Platform and Submission Lead**
  - Own packaging, OpenEnv migration, Space deployment, trainer wiring, README, and reviewer-facing assets.
- **Developer B - Arena and Reward Lead**
  - Own task schema freeze, Fog-of-War, drift engine, verifier/rubrics, synthetic data, and trajectory generation.
- **Critical path for today**
  - Developer A: `0.2 -> 0.3 -> 1.1`
  - Developer B: `0.1 -> 3.1` template/schema work only
  - Shared gate: once `0.1`, `0.3`, and `1.1` are green, expand into full env mechanics, reward logic, and live training.

- [ ] **Task 0.1: Verifiability Gate**
  - Owner: **Developer B**. Review: **Developer A**.
  - Confirm the environment still cleanly fits Theme #3.1 Professional Tasks.
  - Confirm success can be verified programmatically at every key checkpoint.
  - Freeze the v2 action/observation contract and task schema before parallel implementation starts.
  - Define the easiest non-zero-reward curriculum slice before enabling full drift complexity.
- [ ] **Task 0.2: Packaging Baseline**
  - Owner: **Developer A**. Review: **Developer B**.
  - Validate `openenv.yaml`, base class choice, and Gym-style API shape.
  - Confirm client/server separation for Spaces deployment.
  - Audit MCP tool names against reserved OpenEnv names.
- [ ] **Task 0.3: Early Space Deployment**
  - Owner: **Developer A**. Review: **Developer B**.
  - Deploy a minimal runnable environment stub to a dedicated Round 2 Hugging Face Space before full training.
  - Preferred submission slug: `anirudw/rveda-claim-drift-arena`.
  - Keep `anirudw/rveda` untouched as the Round 1 baseline for before-vs-after comparison.
  - Freeze the reviewer-facing URL and README entry points early.

## Sprint 1: The Arena Foundation
*Focus: Environment migration and core Fog-of-War mechanics.*
*Ref: [Audit: Architecture](../rveda-analysis/prompt-01-architecture.md)*

- [ ] **Task 1.1: OpenEnv Latest-Stable Migration**
  - Owner: **Developer A**. Review: **Developer B**.
  - Update `pyproject.toml` and `requirements.txt`.
  - Target the latest stable OpenEnv release at build time; current local floor is `0.2.3`, but do not hard-code that as "latest".
  - Fix `reset` and `step` method signatures and enable concurrent sessions.
- [ ] **Task 1.2: Integrity Pass**
  - Owner: **Developer A**. Review: **Developer B**.
  - Remove `target_code` leakage from search observations.
  - Implement `last_error` and `invalid_reason` in observations.
  - Add timeout and invalid-action surfaces needed for training diagnostics.
- [ ] **Task 1.3: EHR Fog-of-War**
  - Owner: **Developer B**. Review: **Developer A**.
  - Implement `ehr_modules` dictionary in task schema.
  - Implement `QUERY_EHR` action and logic.
  - Update `MedicalObservation` to include `ehr_map`.
  - Keep the ICD/search backend compact and swappable; the core requirement is a `200+` candidate navigation bank, not full-taxonomy ingestion.

## Sprint 2: The Drift and Reward Engine
*Focus: Policy drift, structured reasoning, and reward robustness.*
*Ref: [Strategy: Innovation](../rveda-analysis/prompt-03-innovation-gap.md)*

- [ ] **Task 2.1: Schema Drift Implementation**
  - Owner: **Developer B**. Review: **Developer A**.
  - Create `PolicyEngine` to manage mid-episode drift events.
  - Implement `CHECK_POLICY` and `VALIDATE_CLAIM_SCHEMA` actions.
  - Add `DriftNotice` to observations.
- [ ] **Task 2.2: REASONING_LOG and RLVR**
  - Owner: **Developer B**. Review: **Developer A**.
  - Define `ReasoningLog` and `EvidenceSnippet` models.
  - Implement `verify_reasoning_log`.
  - Keep grounding mandatory before submit.
- [ ] **Task 2.3: Composable Reward Rubrics**
  - Owner: **Developer B**. Review: **Developer A**.
  - Split reward into independent verifier/rubric checks for correctness, grounding, schema compliance, format validity, and anti-cheat behavior.
  - Keep process rewards capped so they cannot rescue wrong terminal answers.
  - Make exact-code correctness dominate family-level matches; family matches should be useful shaping, not a locally optimal terminal policy.
  - Penalize duplicate or excessive search/query behavior and long loopy trajectories.
  - Tie drift adaptation to submission validity and schema outcome, not just whether `CHECK_POLICY` was called.
  - Emit per-rubric metrics for later plots and ablations.

## Sprint 3: Scaling and Training
*Focus: Data generation, curriculum, and GRPO pipeline.*
*Ref: [Strategy: Data Scaling](../rveda-analysis/prompt-04-data-strategy.md)*

- [ ] **Task 3.1: Synthetic Case Generator**
  - Owner: **Developer B**. Review: **Developer A**.
  - Implement `generate_cases.py` using the 10-module template.
  - Build `icd10_expanded.json` (200+ rows).
  - Start only after `Task 0.1` freezes the schema; use the `200+` candidate bank as the default navigation substrate, not the full ICD taxonomy.
  - Create easy-to-hard slices for curriculum training.
- [ ] **Task 3.2: Trajectory Collection**
  - Owner: **Developer B**. Review: **Developer A**.
  - Create `generate_trajectories.py` for teacher/SFT samples.
  - Export `trajectories_sft.jsonl`.
  - Preserve enough trace detail to debug reward hacking and schema failures.
- [ ] **Task 3.3: Minimal TRL/Unsloth Runner**
  - Owner: **Developer A**. Review: **Developer B**.
  - Set up a minimal `GRPOTrainer` script or notebook with the live v2 reward logic.
  - Start as soon as `Task 0.3` and `Task 1.1` are green; do not wait for the full data scale-up before proving live environment connectivity.
  - Configure Unsloth fast-LoRA kernels for Qwen 2.5 7B/14B.
  - Keep the notebook rerunnable in a Colab-like environment.
- [ ] **Task 3.4: Training Observability**
  - Owner: **Developer A**. Review: **Developer B**.
  - Save baseline-vs-trained comparisons.
  - Plot total reward plus rubric sub-metrics with labeled axes.
  - Track Grounding F1, Drift Adaptation Rate, Search-to-Submission Ratio, timeout frequency, and schema validation pass rate.

## Sprint 4: The Finale Pitch
*Focus: README, demo assets, and performance storytelling.*
*Ref: [Independent Solution Analysis](../../sources/report-rveda-solution-analysis.md)*

- [ ] **Task 4.1: Reviewer-Facing README**
  - Owner: **Developer A**. Review: **Developer B**.
  - Make the README the landing page for judges.
  - Link the dedicated Round 2 Hugging Face Space, training notebook, plots, and short video/blog/deck.
  - Link the Round 1 archived Space as the baseline reference, not as the primary deployment.
  - Explain problem, environment loop, results, and why the task matters in under five minutes of reading.
- [ ] **Task 4.2: Caution Dashboard**
  - Owner: **Developer A**. Review: **Developer B**.
  - Plot Grounding F1 vs. Accuracy.
  - Plot Drift Adaptation Rate and schema pass rate.
  - Save static image assets or stable W&B links for every claimed training run.
- [ ] **Task 4.3: Video or Mini-Blog Production**
  - Owner: **Developer A**. Review: **Developer B**.
  - Follow the two-minute "Revenue Cycle Drift Arena" story.
  - Highlight before-vs-after adaptation, not just architecture diagrams.
  - Keep large media out of the environment repo and link externally from the README.

## Stretch Goals (Non-Blocking for Round 2 Compliance)
- [ ] **Stretch S.1: Swappable Taxonomy Backend**
  - Evaluate SQLite FTS or library-backed ICD lookup only if the `200+` candidate bank becomes a bottleneck.
- [ ] **Stretch S.2: External Held-Out Validation**
  - Add a small held-out set of de-identified real notes after the core environment, training proof, and submission assets are stable.
