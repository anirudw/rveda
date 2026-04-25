# Rveda V2 Synthetic Case Generator Plan

## 1. Purpose of Task 3.1

Task 3.1 prepares the synthetic case generation strategy for Rveda V2. This document is planning and schema work only. It does not implement `generate_cases.py`, expand the ICD bank, replace `tasks.json`, or change runtime environment behavior.

The goal is to define how future generated cases will follow the frozen V2 contract from `docs/rveda-v2-contract.md` while supporting curriculum training from easy drift-disabled episodes to harder drift-enabled revenue-cycle scenarios.

## 2. Relationship to the V2 Contract

Generated cases must conform to the Task 0.1 schema:

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

The generator must produce cases that can be verified through the frozen action set:

- `SEARCH_ICD`
- `QUERY_EHR`
- `CHECK_POLICY`
- `VALIDATE_CLAIM_SCHEMA`
- `REASONING_LOG`
- `SUBMIT_CLAIM`

Every generated case should include enough metadata for deterministic checks of evidence reveal, code search, policy validation, reasoning grounding, and terminal claim correctness.

## 3. Round 2 Data Compliance

Generated data must support the updated Round 2 instructions, where judges expect an OpenEnv-hosted environment, a rerunnable Unsloth or HF TRL training script, evidence of reward improvement, coherent rubrics, and a readable README/story.

Compliance rules for generated cases:

- Cases must be fully synthetic. Do not include real patient records, real identifiers, real dates of birth, real medical record numbers, or other PHI.
- Cases must train interaction with the live environment, not replace the environment with a static offline dataset. Generated tasks are the environment substrate; training should still call `reset`, `step`, and `state`.
- Each case must expose a partially observable world. Do not put all decisive evidence in an initial visible note.
- Each case must require tool use across the V2 action set, especially `QUERY_EHR`, `SEARCH_ICD`, `CHECK_POLICY`, `REASONING_LOG`, and `SUBMIT_CLAIM`.
- Each case must include machine-checkable reward signals so progress can be plotted, not only a final correct/incorrect label.
- Cases must include enough rubric metadata to compute grounding, schema compliance, format validity, process discipline, and terminal correctness.
- Cases must be small enough to keep the Hugging Face Space repository lightweight. Large videos or bulky artifacts should not be stored with generated data.
- The generated corpus should support a baseline-vs-trained comparison with readable metrics such as total reward, grounding score, schema pass rate, search-to-submission ratio, drift adaptation rate, invalid-action rate, and timeout rate.
- The dataset should include curriculum splits so early training can get non-zero reward before harder long-horizon or drift-enabled tasks.
- Generated task files should be deterministic and reproducible from a seed so training plots and demo claims can be audited.

## 4. 10-Module EHR Template

The full V2 task template should reserve ten EHR modules, even when easy cases activate only one or two. This keeps case files structurally consistent across curriculum levels.

Recommended module keys:

1. `encounter_note`: visit summary, chief complaint, assessment, and plan.
2. `vitals`: BMI, blood pressure, heart rate, temperature, oxygen saturation.
3. `labs`: lab results, timestamps, abnormal flags, confirmatory diagnostics.
4. `medications`: active meds, discontinued meds, relevant treatment context.
5. `problem_list`: prior diagnoses and unresolved chronic conditions.
6. `procedures`: procedures, imaging, EKGs, or interventions.
7. `history`: past medical, family, and social history.
8. `billing_history`: prior claims, denials, duplicate billing risks.
9. `payer_policy`: payer-specific coding, documentation, and schema rules.
10. `clinician_messages`: follow-up notes, clarifications, or conflicting statements.

Each module should have a stable shape:

```json
{
  "status": "closed",
  "query_budget": 1,
  "evidence": [
    {
      "evidence_id": "ev_example_001",
      "text": "Evidence snippet text.",
      "supports_codes": ["E66.3"],
      "contradicts_codes": [],
      "required_for_grounding": true
    }
  ]
}
```

## 5. V2 Task JSON Template

Future generated tasks should use this expanded template:

```json
{
  "task_id": "v2_easy_example_001",
  "difficulty": "easy",
  "target_code": "E66.3",
  "claim_schema_version": "v1",
  "ehr_modules": {
    "encounter_note": {
      "status": "closed",
      "query_budget": 1,
      "evidence": []
    }
  },
  "target_evidence": ["ev_example_001"],
  "policy_rules": {
    "version": "payer_policy_v1",
    "active_schema_version": "v1",
    "rules": [],
    "claim_schema": {
      "version": "v1",
      "required_fields": [
        "diagnosis_code",
        "schema_version",
        "evidence_ids",
        "reasoning_log_id"
      ]
    }
  },
  "drift": {
    "enabled": false,
    "trigger_step": null,
    "from_schema_version": "v1",
    "to_schema_version": null
  },
  "verification_checkpoints": []
}
```

Easy cases may include a subset of modules. Medium and hard cases should include all ten module keys, with irrelevant or conflicting modules used to test search discipline and grounding.

## 6. Evidence Snippet Format

Evidence snippets are the atomic units used for EHR reveal and grounding verification.

Required fields:

- `evidence_id`: stable unique ID within the task.
- `text`: human-readable evidence.
- `supports_codes`: list of ICD codes the evidence supports.
- `contradicts_codes`: list of ICD codes the evidence argues against.
- `required_for_grounding`: boolean indicating whether the evidence should be cited for full grounding credit.

Optional future fields:

- `module`: redundant module name for easier validation after flattening.
- `timestamp`: clinical or billing timestamp.
- `source_type`: note, lab, vital, procedure, policy, or billing artifact.
- `confidence`: synthetic confidence score for noisy cases.

Validation rule: every ID in `target_evidence` must exist in exactly one module evidence list and should have `required_for_grounding = true`.

## 7. Policy Rule Format

Policy rules should be structured enough to support `CHECK_POLICY`, `VALIDATE_CLAIM_SCHEMA`, and later drift.

Recommended rule shape:

```json
{
  "rule_id": "policy_v1_evidence_required",
  "description": "A submitted diagnosis claim must cite at least one revealed evidence snippet.",
  "applies_to_codes": ["E66.3"],
  "required_fields": ["diagnosis_code", "schema_version", "evidence_ids"],
  "validation_type": "required_field"
}
```

Recommended `policy_rules` object:

```json
{
  "version": "payer_policy_v1",
  "active_schema_version": "v1",
  "rules": [],
  "claim_schema": {
    "version": "v1",
    "required_fields": [
      "diagnosis_code",
      "schema_version",
      "evidence_ids",
      "reasoning_log_id"
    ]
  }
}
```

## 8. Drift-Disabled Easy Curriculum Slice

Easy cases should make non-zero reward reachable without requiring policy drift.

Easy slice rules:

- `difficulty = "easy"`
- `drift.enabled = false`
- one target ICD code
- one or two active EHR modules
- one required evidence snippet
- no contradictory evidence
- one simple policy rule
- natural ICD query should return the target code or family

Expected successful path:

1. `QUERY_EHR` reveals required evidence.
2. `SEARCH_ICD` returns the target code or family.
3. `CHECK_POLICY` reveals schema version `v1`.
4. `VALIDATE_CLAIM_SCHEMA` accepts a valid draft claim.
5. `REASONING_LOG` cites the required evidence.
6. `SUBMIT_CLAIM` submits exact target code with correct schema and evidence.

## 9. Future Medium And Hard Curriculum Slices

Medium cases:

- multiple EHR modules with one or two relevant snippets
- at least one plausible distractor ICD family
- one policy requirement that can fail validation
- optional drift disabled at first, then enabled after runtime support exists

Hard cases:

- all ten EHR modules present
- multiple relevant and conflicting snippets
- larger candidate bank with near-neighbor codes
- drift enabled after `CHECK_POLICY` or mid-episode step threshold
- final claim must adapt to the post-drift schema

Hard examples should test whether the agent avoids aggressive early submission and can reconcile evidence, policy, and schema changes before terminal action.

## 10. ICD Candidate Bank Requirement

Task 3.1 eventually requires an expanded ICD bank with at least 200 rows. The current `icd10_mock.json` has only a compact mock set and should remain unchanged during this planning step.

Future `icd10_expanded.json` should include:

- target codes used by generated cases
- same-family distractors
- clinically plausible near-neighbors
- unrelated distractors for search discipline
- `short_desc`, `long_desc`, and `excludes` for every code

The generator should verify that every `target_code` appears in the candidate bank and that natural queries for the case can retrieve the target code or family.

## 11. Validation Rules For Generated Cases

The future generator should reject any generated case that violates these rules:

- `task_id` is unique.
- `difficulty` is one of `easy`, `medium`, or `hard`.
- case content is synthetic and contains no PHI-like identifiers.
- `target_code` exists in the ICD candidate bank.
- `claim_schema_version` matches `policy_rules.active_schema_version` when drift is disabled.
- `ehr_modules` is non-empty.
- each module has `status`, `query_budget`, and `evidence`.
- every evidence item has a unique `evidence_id`.
- every `target_evidence` ID exists exactly once.
- every required evidence item supports the `target_code` or target family.
- every policy rule has a stable `rule_id`.
- drift-disabled cases use explicit null drift targets.
- drift-enabled cases define trigger step, source schema, target schema, and changed required fields.
- `verification_checkpoints` covers all six frozen V2 actions.
- reward metadata is sufficient to compute per-rubric metrics and training plots.
- every case can be exercised through the OpenEnv loop rather than only as an offline supervised sample.
- generated outputs are deterministic from a recorded seed.

## 12. Planned Outputs

Future Task 3.1 implementation should produce:

- `generate_cases.py`: deterministic synthetic case generator.
- `icd10_expanded.json`: 200+ row ICD navigation bank.
- generated V2 task dataset, for example `tasks_v2.json`.
- optional validation report summarizing generated case counts, difficulty mix, target-code coverage, schema validity, PHI-safety checks, and reward-metric coverage.

These files are not created in this planning step.

## 13. Explicit Deferred Work

Deferred beyond this template/schema plan:

- implementing `generate_cases.py`
- generating 200+ ICD rows
- replacing `tasks.json`
- modifying `models.py`
- modifying `server/rveda_environment.py`
- implementing `QUERY_EHR`
- implementing policy drift
- implementing reasoning-log verification
- implementing reward rubrics
- generating teacher trajectories
- wiring the generated dataset into training
