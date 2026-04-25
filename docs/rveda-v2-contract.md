# Rveda V2 Contract: Task 0.1 Verifiability Gate

## 1. Purpose of Task 0.1

Task 0.1 freezes the verifiable contract for Rveda V2 before runtime mechanics are implemented.

Rveda V2, "Revenue Cycle Drift Arena," upgrades the Round 1 medical-coding benchmark into a partially observable professional-task arena. The agent must operate through explicit tools, reveal hidden information, check policy constraints, produce grounded reasoning, and submit a structured claim.

This document is a contract/specification only. It does not claim that the V2 mechanics already exist in the current repository.

Task 0.1 deliverables are:

- Confirm the V2 environment fits OpenEnv Round 2 Theme #3.1 Professional Tasks.
- Freeze the V2 action schema.
- Freeze the V2 observation schema.
- Freeze the V2 task schema.
- Define programmatic verifiability checkpoints.
- Define the easiest non-zero-reward curriculum slice.
- Identify work deferred to later implementation tasks.

## 2. Theme #3.1 Professional Tasks Fit

Rveda V2 fits Theme #3.1 Professional Tasks because it models a realistic revenue-cycle workflow rather than a single-label classification task.

The professional workflow requires the agent to:

- query hidden EHR modules for relevant clinical evidence,
- navigate ICD diagnosis candidates,
- check insurance policy and schema rules,
- validate whether a structured claim matches the active schema,
- write a grounded reasoning log with evidence citations,
- submit a structured claim payload,
- later adapt when policy or claim schema rules drift during an episode.

The task is verifiable because each stage produces structured state changes and rubric metrics: evidence revealed, codes searched, policy checked, schema validated, reasoning grounded, and claim submitted.

## 3. Hackathon Constraints

Task 0.1 is a guardrail task. It must not implement full V2 runtime behavior.

Constraints:

- Keep the current Round 1 environment intact until the V2 contract is frozen.
- Do not modify `server/rveda_environment.py` for Task 0.1.
- Do not modify `models.py` for Task 0.1.
- Do not modify `tasks.json` for Task 0.1.
- Do not modify `icd10_mock.json` for Task 0.1.
- Do not claim V2 runtime mechanics exist before they are implemented.
- Keep actions and observations strict-JSON compatible.
- Preserve OpenEnv compatibility for later `reset`, `step`, and `state` implementation work.
- Keep process rewards capped so they cannot rescue an incorrect terminal claim.
- Make all reward components inspectable for training plots and ablations.

## 4. Frozen V2 Action Schema

All actions use a top-level `action_type` string. Actions with extra inputs must place structured data in named fields or `payload`.

### `SEARCH_ICD`

Search the ICD candidate bank.

```json
{
  "action_type": "SEARCH_ICD",
  "query": "string",
  "page": 1
}
```

Fields:

- `query`: required string. Clinical keyword or diagnosis phrase.
- `page`: optional integer, default `1`. Used for paginated candidate banks.

### `QUERY_EHR`

Reveal evidence from one hidden EHR module.

```json
{
  "action_type": "QUERY_EHR",
  "module": "string",
  "query": "string"
}
```

Fields:

- `module`: required string. Must reference a module key from the task `ehr_modules`.
- `query`: required string. Search term or focused request within the module.

### `CHECK_POLICY`

Reveal current insurance policy and claim schema rules.

```json
{
  "action_type": "CHECK_POLICY"
}
```

Fields:

- No additional required fields.

### `VALIDATE_CLAIM_SCHEMA`

Validate a draft claim against the currently active schema without ending the episode.

```json
{
  "action_type": "VALIDATE_CLAIM_SCHEMA",
  "payload": {
    "diagnosis_code": "string",
    "schema_version": "string",
    "evidence_ids": ["string"],
    "policy_attestations": ["string"]
  }
}
```

Fields:

- `payload.diagnosis_code`: required string.
- `payload.schema_version`: required string.
- `payload.evidence_ids`: required list of evidence IDs.
- `payload.policy_attestations`: optional list of policy rule IDs or attestations.

### `REASONING_LOG`

Submit a grounded reasoning record before final claim submission.

```json
{
  "action_type": "REASONING_LOG",
  "payload": {
    "candidate_code": "string",
    "rationale": "string",
    "evidence_ids": ["string"],
    "policy_rule_ids": ["string"]
  }
}
```

Fields:

- `payload.candidate_code`: required string.
- `payload.rationale`: required string.
- `payload.evidence_ids`: required list of revealed evidence IDs.
- `payload.policy_rule_ids`: optional list of checked policy rule IDs.

### `SUBMIT_CLAIM`

Submit the terminal structured claim.

```json
{
  "action_type": "SUBMIT_CLAIM",
  "payload": {
    "diagnosis_code": "string",
    "schema_version": "string",
    "evidence_ids": ["string"],
    "reasoning_log_id": "string",
    "policy_attestations": ["string"]
  }
}
```

Fields:

- `payload.diagnosis_code`: required string.
- `payload.schema_version`: required string.
- `payload.evidence_ids`: required list of revealed evidence IDs.
- `payload.reasoning_log_id`: required string or environment-provided ID from the latest valid `REASONING_LOG`.
- `payload.policy_attestations`: optional list of policy rule IDs or attestations.

## 5. Frozen V2 Observation Schema

The V2 observation must support partial observability and verifier diagnostics.

```json
{
  "ehr_map": {},
  "search_results": [],
  "revealed_evidence": [],
  "policy_state": {},
  "drift_notice": null,
  "last_error": null,
  "invalid_reason": null,
  "grading": {},
  "reward_metrics": {},
  "done": false,
  "reward": 0.0
}
```

Fields:

- `ehr_map`: object keyed by EHR module name. Each value includes module status such as `closed`, `open`, or `exhausted`, plus remaining query budget when applicable.
- `search_results`: list of ICD candidate summaries. Each item should include at least `code` and `short_desc`; pagination metadata may be included.
- `revealed_evidence`: list of evidence snippets the agent has unlocked. Each item must include a stable `evidence_id`, `module`, and snippet text or structured value.
- `policy_state`: object describing visible policy rules, active claim schema version, and whether policy has been checked.
- `drift_notice`: null or structured notice that policy/schema drift occurred.
- `last_error`: null or machine-readable error category from the previous step.
- `invalid_reason`: null or human-readable reason the previous action was invalid.
- `grading`: structured trace for the current step and episode state.
- `reward_metrics`: rubric-level numeric metrics.
- `done`: boolean terminal flag.
- `reward`: numeric reward for the current step.

Required `reward_metrics` keys:

- `terminal_correctness`
- `evidence_grounding`
- `schema_compliance`
- `format_validity`
- `process_discipline`
- `drift_adaptation`
- `final`

## 6. Frozen V2 Task Schema

Each V2 task must be a structured object.

```json
{
  "task_id": "string",
  "difficulty": "easy",
  "target_code": "string",
  "claim_schema_version": "v1",
  "ehr_modules": {},
  "target_evidence": [],
  "policy_rules": {},
  "drift": {},
  "verification_checkpoints": []
}
```

Fields:

- `task_id`: required stable task identifier.
- `difficulty`: required difficulty label, such as `easy`, `medium`, or `hard`.
- `target_code`: required ICD target code for verifier use only.
- `claim_schema_version`: required starting schema version.
- `ehr_modules`: required object keyed by module name. Each module contains hidden evidence snippets and query limits.
- `target_evidence`: required list of evidence IDs needed to ground the correct claim.
- `policy_rules`: required object containing schema and payer rules.
- `drift`: required object. For no-drift tasks, use an explicit disabled config.
- `verification_checkpoints`: required list of machine-checkable checkpoint definitions for Task 0.1 examples and verifier tests.

Example no-drift config:

```json
{
  "enabled": false,
  "trigger_step": null,
  "from_schema_version": "v1",
  "to_schema_version": null
}
```

## 7. Programmatic Verifiability Checkpoints

Each checkpoint must be assertable from task state, action payloads, observations, and grading metadata.

| Checkpoint | Programmatic verification |
| --- | --- |
| EHR query validity | `QUERY_EHR.module` exists in `task.ehr_modules`; module budget remains; observation updates `ehr_map`. |
| Evidence reveal | Returned evidence IDs exist in the selected module and are appended to `revealed_evidence`. |
| ICD search | `SEARCH_ICD.query` produces candidate summaries from the ICD bank without exposing `target_code`. |
| Policy check | `CHECK_POLICY` updates `policy_state.checked = true` and reveals active schema/policy rules. |
| Schema validation | `VALIDATE_CLAIM_SCHEMA.payload` is checked against active `claim_schema_version`; result is surfaced through `last_error`, `invalid_reason`, `policy_state`, or `grading`. |
| Reasoning log | `REASONING_LOG.payload.evidence_ids` must cite already revealed evidence; grounding metrics are computed. |
| Structured claim submit | `SUBMIT_CLAIM.payload` must include diagnosis code, schema version, evidence IDs, and reasoning log reference. |
| Terminal correctness | Submitted diagnosis code is compared to `target_code`; exact match dominates family match. |
| Grounding | Required `target_evidence` coverage is compared to cited evidence IDs. |
| Drift adaptation | When drift is enabled, final claim schema and policy attestations must match the post-drift active schema. |
| Timeout/process discipline | Step count, duplicate actions, invalid actions, and loops are tracked in `reward_metrics.process_discipline`. |

## 8. Minimal Non-Zero-Reward Curriculum Slice

The easiest initial curriculum slice should make non-zero reward reachable before full drift complexity.

Configuration:

- `difficulty`: `easy`
- `drift.enabled`: `false`
- one target ICD code,
- one EHR module with one directly relevant evidence snippet,
- one starting claim schema version, such as `v1`,
- one simple policy rule,
- small ICD candidate bank sufficient to return the target code from a natural query,
- no mid-episode policy/schema mutation.

Minimum rewarding trajectory:

1. `QUERY_EHR` reveals the target evidence.
2. `SEARCH_ICD` returns the target code or target family.
3. `CHECK_POLICY` reveals the active schema.
4. `VALIDATE_CLAIM_SCHEMA` confirms a draft claim shape or returns a correctable error.
5. `REASONING_LOG` cites the revealed target evidence.
6. `SUBMIT_CLAIM` submits the exact target code with the correct schema version and evidence IDs.

Reward expectations:

- Small process reward for valid evidence reveal.
- Small process reward for useful ICD search.
- Small process reward for checking policy.
- Small process reward for valid draft schema validation.
- Grounding reward for citing revealed target evidence.
- Terminal reward dominated by exact-code correctness and schema compliance.
- Wrong terminal claims must not be rescued by process rewards.

## 9. Deferred Work

The following work is explicitly deferred beyond Task 0.1:

- Implementing V2 runtime mechanics in `server/rveda_environment.py`.
- Updating Pydantic/OpenEnv models in `models.py`.
- Replacing or migrating `tasks.json` to the V2 task format.
- Expanding `icd10_mock.json` or creating a 200+ code candidate bank.
- Implementing full Fog-of-War behavior.
- Implementing EHR module state, query budgets, and evidence reveal logic.
- Implementing policy engine and drift engine.
- Implementing `VALIDATE_CLAIM_SCHEMA`.
- Implementing `REASONING_LOG` persistence and grounding verification.
- Implementing composable reward rubrics.
- Generating synthetic cases.
- Generating trajectory datasets.
- Building training scripts, plots, or deployment assets.

## 10. Developer A Handoff

Developer A can use this contract to proceed with packaging and deployment guardrails without depending on unfinished V2 mechanics.

Handoff items:

- Preserve Round 1 runtime behavior until V2 implementation tasks begin.
- Treat this document as the frozen Task 0.1 schema contract.
- Validate that future OpenEnv packaging can support the six V2 actions and observation fields listed above.
- Confirm that the server/client boundary can carry strict JSON action and observation payloads.
- Keep a dedicated Round 2 deployment target separate from the archived Round 1 baseline.
- Do not require V2 mechanics to exist before minimal Space/deployment scaffolding is reviewed.
