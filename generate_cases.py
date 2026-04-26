"""Deterministic synthetic V2 case generator for Sprint 3 Task 3.1."""

from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

MODULE_KEYS = [
    "encounter_note",
    "vitals",
    "labs",
    "medications",
    "problem_list",
    "procedures",
    "history",
    "billing_history",
    "payer_policy",
    "clinician_messages",
]

V1_REQUIRED_FIELDS = [
    "diagnosis_code",
    "schema_version",
    "evidence_ids",
    "reasoning_log_id",
]
V2_REQUIRED_FIELDS = [*V1_REQUIRED_FIELDS, "policy_attestations"]

DEFAULT_SPLIT_PLAN = {
    "train": {"easy": 2, "medium": 2, "hard": 2},
    "eval": {"easy": 1, "medium": 1, "hard": 1},
    "smoke": {"easy": 1, "medium": 1, "hard": 1},
}

CASE_BLUEPRINTS: dict[str, list[dict[str, Any]]] = {
    "easy": [
        {
            "slug": "overweight_counseling",
            "target_code": "E66.3",
            "visible_note": (
                "Routine outpatient follow-up with diet and exercise counseling documented, "
                "but the decisive coding evidence remains in the hidden chart."
            ),
            "search_terms": ["overweight bmi counseling", "bmi 27 weight management"],
            "target_evidence": [
                {
                    "module": "encounter_note",
                    "text": (
                        "Routine checkup documents BMI 27 with diet and exercise counseling "
                        "for weight management."
                    ),
                    "source_type": "note",
                }
            ],
            "distractors": [
                {
                    "module": "problem_list",
                    "text": "Problem list notes prior thyroid screening without active disease.",
                    "supports_codes": ["E03.9"],
                    "source_type": "problem_list",
                }
            ],
        },
        {
            "slug": "hashimoto_confirmation",
            "target_code": "E06.3",
            "visible_note": (
                "Endocrinology follow-up for fatigue and thyroid symptoms. "
                "Hidden labs and problem list determine the most specific code."
            ),
            "search_terms": ["hashimoto tpo antibodies", "autoimmune thyroiditis"],
            "target_evidence": [
                {
                    "module": "labs",
                    "text": "Lab review shows elevated TPO antibodies consistent with Hashimoto disease.",
                    "source_type": "lab",
                }
            ],
            "distractors": [
                {
                    "module": "history",
                    "text": "Family history includes unspecified hypothyroidism in a parent.",
                    "supports_codes": ["E03.9"],
                    "source_type": "history",
                }
            ],
        },
        {
            "slug": "hypothyroid_unspecified",
            "target_code": "E03.9",
            "visible_note": (
                "Primary care visit for chronic fatigue; specific autoimmune confirmation is absent "
                "unless the agent over-reads distractors."
            ),
            "search_terms": ["hypothyroidism unspecified", "fatigue thyroid medication"],
            "target_evidence": [
                {
                    "module": "medications",
                    "text": "Medication list confirms levothyroxine continuation for unspecified hypothyroidism.",
                    "source_type": "medication",
                }
            ],
            "distractors": [
                {
                    "module": "labs",
                    "text": "TSH is mildly elevated, but no antibody workup was completed.",
                    "supports_codes": ["E03.9"],
                    "source_type": "lab",
                }
            ],
        },
    ],
    "medium": [
        {
            "slug": "diabetes_hyperglycemia_drift",
            "target_code": "E11.65",
            "starting_schema_version": "v1",
            "drift_to_schema_version": "v2",
            "visible_note": (
                "Diabetes follow-up with medication reconciliation completed. "
                "The glucose evidence and post-drift claim requirements must be checked explicitly."
            ),
            "search_terms": ["type 2 diabetes hyperglycemia", "poorly controlled blood sugar"],
            "target_evidence": [
                {
                    "module": "labs",
                    "text": "Fasting glucose remains elevated despite treatment, documenting hyperglycemia.",
                    "source_type": "lab",
                },
                {
                    "module": "medications",
                    "text": "Medication review notes intensified diabetes regimen for uncontrolled sugars.",
                    "source_type": "medication",
                },
            ],
            "distractors": [
                {
                    "module": "billing_history",
                    "text": "Prior clean claim used schema v1 before the payer updated attestation rules.",
                    "supports_codes": ["E11.9"],
                    "source_type": "billing",
                }
            ],
        },
        {
            "slug": "ckd_stage_three",
            "target_code": "N18.3",
            "starting_schema_version": "v2",
            "visible_note": (
                "Nephrology follow-up references chronic kidney disease progression, "
                "but stage confirmation is hidden in the chart."
            ),
            "search_terms": ["chronic kidney disease stage 3", "moderate ckd"],
            "target_evidence": [
                {
                    "module": "labs",
                    "text": "Metabolic panel and nephrology assessment document chronic kidney disease stage 3.",
                    "source_type": "lab",
                },
                {
                    "module": "problem_list",
                    "text": "Problem list marks CKD stage 3 as an active chronic condition.",
                    "source_type": "problem_list",
                },
            ],
            "distractors": [
                {
                    "module": "history",
                    "text": "History mentions prior acute kidney injury now resolved.",
                    "supports_codes": ["N18.3"],
                    "source_type": "history",
                }
            ],
        },
        {
            "slug": "chest_pain_observation",
            "target_code": "R07.9",
            "starting_schema_version": "v1",
            "visible_note": (
                "Emergency observation visit for chest discomfort with no hidden cardiac infarction evidence."
            ),
            "search_terms": ["chest pain unspecified", "observation chest discomfort"],
            "target_evidence": [
                {
                    "module": "encounter_note",
                    "text": "Observation note documents chest pain without definitive infarction diagnosis.",
                    "source_type": "note",
                },
                {
                    "module": "procedures",
                    "text": "Initial ECG review is nondiagnostic for acute MI.",
                    "source_type": "procedure",
                },
            ],
            "distractors": [
                {
                    "module": "labs",
                    "text": "Serial troponins remain within normal range during observation.",
                    "supports_codes": ["R07.9"],
                    "source_type": "lab",
                }
            ],
        },
    ],
    "hard": [
        {
            "slug": "acute_mi_drift",
            "target_code": "I21.9",
            "starting_schema_version": "v1",
            "drift_to_schema_version": "v2",
            "visible_note": (
                "Emergency cardiac admission requires coordinated evidence, policy re-check, "
                "and a post-drift valid claim."
            ),
            "search_terms": ["acute myocardial infarction", "heart attack troponin ekg"],
            "target_evidence": [
                {
                    "module": "encounter_note",
                    "text": "ER note documents crushing chest pain radiating to the left arm.",
                    "source_type": "note",
                },
                {
                    "module": "labs",
                    "text": "Troponin levels are markedly elevated, supporting acute myocardial infarction.",
                    "source_type": "lab",
                },
                {
                    "module": "procedures",
                    "text": "EKG interpretation confirms an acute myocardial infarction pattern.",
                    "source_type": "procedure",
                },
            ],
            "distractors": [
                {
                    "module": "billing_history",
                    "text": "Previous chest-pain observation claim used schema v1 without attestation.",
                    "supports_codes": ["R07.9"],
                    "source_type": "billing",
                },
                {
                    "module": "history",
                    "text": "History includes prior reflux symptoms unrelated to the current event.",
                    "supports_codes": ["R07.9"],
                    "source_type": "history",
                },
            ],
        },
        {
            "slug": "diabetes_ckd_combo_drift",
            "target_code": "E11.22",
            "starting_schema_version": "v1",
            "drift_to_schema_version": "v2",
            "visible_note": (
                "Complex diabetes follow-up with renal involvement and multiple plausible near-neighbor codes."
            ),
            "search_terms": ["diabetes chronic kidney disease", "diabetic ckd"],
            "target_evidence": [
                {
                    "module": "labs",
                    "text": "Assessment links diabetic kidney disease to the patient's type 2 diabetes.",
                    "source_type": "lab",
                },
                {
                    "module": "problem_list",
                    "text": "Problem list explicitly states diabetic chronic kidney disease.",
                    "source_type": "problem_list",
                },
            ],
            "distractors": [
                {
                    "module": "history",
                    "text": "Separate note lists chronic kidney disease stage 3 as a coexisting condition.",
                    "supports_codes": ["N18.3"],
                    "source_type": "history",
                },
                {
                    "module": "medications",
                    "text": "Medication profile includes intensified diabetes treatment for chronic control issues.",
                    "supports_codes": ["E11.65"],
                    "source_type": "medication",
                },
            ],
        },
        {
            "slug": "morbid_obesity_policy_drift",
            "target_code": "E66.01",
            "starting_schema_version": "v1",
            "drift_to_schema_version": "v2",
            "visible_note": (
                "Longitudinal obesity-management case with payer-rule drift and competing obesity-family codes."
            ),
            "search_terms": ["morbid obesity excess calories", "severe obesity counseling"],
            "target_evidence": [
                {
                    "module": "vitals",
                    "text": "Vitals show BMI 41 documenting morbid obesity due to excess calories.",
                    "source_type": "vital",
                },
                {
                    "module": "encounter_note",
                    "text": "Clinician assessment explicitly documents severe obesity due to excess calories.",
                    "source_type": "note",
                },
            ],
            "distractors": [
                {
                    "module": "problem_list",
                    "text": "Legacy problem list still contains a generic overweight entry from an earlier visit.",
                    "supports_codes": ["E66.3"],
                    "source_type": "problem_list",
                },
                {
                    "module": "clinician_messages",
                    "text": "Message confirms updated payer attestation rules for obesity claims after drift.",
                    "supports_codes": ["E66.01"],
                    "source_type": "message",
                },
            ],
        },
    ],
}


def _family(code: str) -> str:
    return code[:3]


def _load_icd_index(bank_path: Path) -> dict[str, dict[str, Any]]:
    rows = json.loads(bank_path.read_text(encoding="utf-8"))
    return {str(row["code"]): dict(row) for row in rows}


def _required_fields_for_schema(schema_version: str) -> list[str]:
    return list(V2_REQUIRED_FIELDS if schema_version == "v2" else V1_REQUIRED_FIELDS)


def _module_shell(query_budget: int = 0) -> dict[str, Any]:
    return {"status": "closed", "query_budget": query_budget, "evidence": []}


def _blank_modules() -> dict[str, dict[str, Any]]:
    return {module_name: _module_shell() for module_name in MODULE_KEYS}


def _build_evidence(
    *,
    evidence_id: str,
    module: str,
    text: str,
    supports_codes: list[str],
    required_for_grounding: bool,
    source_type: str,
    contradicts_codes: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "evidence_id": evidence_id,
        "text": text,
        "supports_codes": supports_codes,
        "contradicts_codes": contradicts_codes or [],
        "required_for_grounding": required_for_grounding,
        "source_type": source_type,
        "module": module,
    }


def _build_policy_rules(
    *,
    target_code: str,
    schema_version: str,
    drift_enabled: bool,
) -> dict[str, Any]:
    rules = [
        {
            "rule_id": "policy_evidence_required",
            "description": "A submitted diagnosis claim must cite revealed evidence snippets.",
            "applies_to_codes": [target_code],
            "required_fields": ["evidence_ids"],
            "validation_type": "required_field",
        },
        {
            "rule_id": "policy_reasoning_required",
            "description": "A submitted diagnosis claim must reference a verified reasoning log.",
            "applies_to_codes": [target_code],
            "required_fields": ["reasoning_log_id"],
            "validation_type": "required_field",
        },
    ]
    if schema_version == "v2" or drift_enabled:
        rules.append(
            {
                "rule_id": "policy_attestation_required",
                "description": "Post-drift claims must include a non-empty policy attestation.",
                "applies_to_codes": [target_code],
                "required_fields": ["policy_attestations"],
                "validation_type": "required_field",
            }
        )

    return {
        "version": f"payer_policy_{schema_version}",
        "active_schema_version": schema_version,
        "rules": rules,
        "claim_schema": {
            "version": schema_version,
            "required_fields": _required_fields_for_schema(schema_version),
        },
    }


def _build_claim_example(
    *,
    target_code: str,
    schema_version: str,
    target_evidence_ids: list[str],
    require_policy_attestation: bool,
) -> dict[str, Any]:
    claim = {
        "diagnosis_code": target_code,
        "schema_version": schema_version,
        "evidence_ids": list(target_evidence_ids),
        "reasoning_log_id": "rl_example",
    }
    if require_policy_attestation:
        claim["policy_attestations"] = ["policy_attestation_required"]
    return claim


def _build_schema_expectations(
    *,
    target_code: str,
    required_schema_version: str,
    required_fields: list[str],
    target_evidence_ids: list[str],
) -> dict[str, Any]:
    require_attestation = "policy_attestations" in required_fields
    valid_claim = _build_claim_example(
        target_code=target_code,
        schema_version=required_schema_version,
        target_evidence_ids=target_evidence_ids,
        require_policy_attestation=require_attestation,
    )
    wrong_version = "v1" if required_schema_version == "v2" else "v2"
    invalid_examples = [
        {
            "label": "wrong_schema_version",
            "claim": {
                **valid_claim,
                "schema_version": wrong_version,
            },
        },
        {
            "label": "missing_reasoning_log_id",
            "claim": {
                key: value
                for key, value in valid_claim.items()
                if key != "reasoning_log_id"
            },
        },
    ]
    if require_attestation:
        invalid_examples.append(
            {
                "label": "missing_policy_attestations",
                "claim": {
                    key: value
                    for key, value in valid_claim.items()
                    if key != "policy_attestations"
                },
            }
        )
    return {
        "required_schema_version": required_schema_version,
        "required_fields": list(required_fields),
        "expected_valid_claim": valid_claim,
        "expected_invalid_claims": invalid_examples,
    }


def _build_drift_config(
    *,
    enabled: bool,
    trigger_step: int | None,
    from_schema_version: str,
    to_schema_version: str | None,
) -> dict[str, Any]:
    if not enabled:
        return {
            "enabled": False,
            "trigger_step": None,
            "from_schema_version": from_schema_version,
            "to_schema_version": None,
            "required_fields": _required_fields_for_schema(from_schema_version),
            "expected_adaptation_action": None,
            "expected_adaptation_criterion": "No adaptation required; schema remains stable.",
        }
    return {
        "enabled": True,
        "trigger_step": trigger_step,
        "from_schema_version": from_schema_version,
        "to_schema_version": to_schema_version,
        "required_fields": _required_fields_for_schema(to_schema_version or from_schema_version),
        "expected_adaptation_action": "VALIDATE_CLAIM_SCHEMA",
        "expected_adaptation_criterion": (
            "After drift, the agent must validate and submit using the post-drift schema version "
            f"{to_schema_version} with all required fields present."
        ),
    }


def _build_checkpoints(
    *,
    target_module: str,
    target_evidence_ids: list[str],
    search_terms: list[str],
    policy_schema_version: str,
    submit_schema_version: str,
    target_code: str,
) -> list[dict[str, Any]]:
    return [
        {
            "checkpoint_id": "reveal_target_evidence",
            "action_type": "QUERY_EHR",
            "required": True,
            "expected_module": target_module,
            "success_criterion": {
                "revealed_evidence_ids": list(target_evidence_ids),
                "ehr_status_after_query": "open_or_exhausted",
            },
        },
        {
            "checkpoint_id": "search_target_found",
            "action_type": "SEARCH",
            "required": True,
            "success_criterion": {
                "recommended_queries": list(search_terms),
                "target_code": target_code,
                "accepted_code_family": _family(target_code),
            },
        },
        {
            "checkpoint_id": "policy_checked",
            "action_type": "CHECK_POLICY",
            "required": True,
            "success_criterion": {
                "policy_checked": True,
                "active_schema_version": policy_schema_version,
            },
        },
        {
            "checkpoint_id": "schema_validated",
            "action_type": "VALIDATE_CLAIM_SCHEMA",
            "required": True,
            "success_criterion": {
                "required_schema_version": submit_schema_version,
                "required_fields_present": True,
            },
        },
        {
            "checkpoint_id": "reasoning_logged",
            "action_type": "REASONING_LOG",
            "required": True,
            "success_criterion": {
                "reasoning_verified": True,
                "cited_evidence_ids": list(target_evidence_ids),
            },
        },
        {
            "checkpoint_id": "final_submit_valid",
            "action_type": "SUBMIT",
            "required": True,
            "success_criterion": {
                "diagnosis_code": target_code,
                "schema_version": submit_schema_version,
                "terminal_correctness": "exact",
            },
        },
    ]


def _case_plan_items() -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for split, plan in DEFAULT_SPLIT_PLAN.items():
        for difficulty in ("easy", "medium", "hard"):
            for _ in range(plan[difficulty]):
                items.append((split, difficulty))
    return items


def _build_task(
    *,
    blueprint: dict[str, Any],
    split: str,
    sequence_index: int,
    rng: random.Random,
    code_index: dict[str, dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    target_code = str(blueprint["target_code"])
    if target_code not in code_index:
        raise ValueError(f"Target code {target_code} is missing from the candidate bank.")

    difficulty = str(blueprint.get("difficulty") or "")
    if not difficulty:
        raise ValueError("Blueprint difficulty is required.")
    starting_schema_version = str(blueprint.get("starting_schema_version", "v1"))
    drift_to_schema_version = blueprint.get("drift_to_schema_version")
    drift_enabled = bool(drift_to_schema_version)
    final_schema_version = str(drift_to_schema_version or starting_schema_version)
    final_required_fields = _required_fields_for_schema(final_schema_version)
    modules = _blank_modules()

    target_evidence_ids: list[str] = []
    for evidence_index, evidence_spec in enumerate(blueprint.get("target_evidence", []), start=1):
        module_name = str(evidence_spec["module"])
        evidence_id = f"ev_{blueprint['slug']}_{evidence_index:02d}"
        modules[module_name]["query_budget"] = max(modules[module_name]["query_budget"], 1)
        modules[module_name]["evidence"].append(
            _build_evidence(
                evidence_id=evidence_id,
                module=module_name,
                text=str(evidence_spec["text"]),
                supports_codes=[target_code],
                required_for_grounding=True,
                source_type=str(evidence_spec.get("source_type", "note")),
            )
        )
        target_evidence_ids.append(evidence_id)

    distractor_specs = list(blueprint.get("distractors", []))
    rng.shuffle(distractor_specs)
    for distractor_index, distractor_spec in enumerate(distractor_specs, start=1):
        module_name = str(distractor_spec["module"])
        support_code = str(distractor_spec.get("supports_codes", [target_code])[0])
        modules[module_name]["query_budget"] = max(modules[module_name]["query_budget"], 1)
        modules[module_name]["evidence"].append(
            _build_evidence(
                evidence_id=f"ev_{blueprint['slug']}_dx_{distractor_index:02d}",
                module=module_name,
                text=str(distractor_spec["text"]),
                supports_codes=[support_code],
                required_for_grounding=False,
                source_type=str(distractor_spec.get("source_type", "note")),
                contradicts_codes=[target_code] if support_code != target_code else [],
            )
        )

    policy_rules = _build_policy_rules(
        target_code=target_code,
        schema_version=starting_schema_version,
        drift_enabled=drift_enabled,
    )
    drift = _build_drift_config(
        enabled=drift_enabled,
        trigger_step=3 if drift_enabled and difficulty != "hard" else 2 if drift_enabled else None,
        from_schema_version=starting_schema_version,
        to_schema_version=str(drift_to_schema_version) if drift_to_schema_version else None,
    )
    task_id = (
        f"v2_{split}_{difficulty}_{blueprint['slug']}_"
        f"{'drift' if drift_enabled else final_schema_version}_{sequence_index:03d}"
    )
    schema_expectations = _build_schema_expectations(
        target_code=target_code,
        required_schema_version=final_schema_version,
        required_fields=final_required_fields,
        target_evidence_ids=target_evidence_ids,
    )
    target_module = str(blueprint["target_evidence"][0]["module"])

    return {
        "task_id": task_id,
        "split": split,
        "split_tags": {
            "split": split,
            "difficulty": difficulty,
            "drift_profile": "drift" if drift_enabled else "no_drift",
            "schema_version": final_schema_version,
        },
        "difficulty": difficulty,
        "patient_note": str(blueprint["visible_note"]),
        "target_code": target_code,
        "claim_schema_version": starting_schema_version,
        "terminal_answer_labels": {
            "target_code": target_code,
            "accepted_code_family": _family(target_code),
            "accepted_alternatives": list(blueprint.get("accepted_alternatives", [])),
        },
        "ehr_modules": modules,
        "target_evidence": list(target_evidence_ids),
        "target_evidence_ids": list(target_evidence_ids),
        "policy_rules": policy_rules,
        "drift": drift,
        "schema_validation_expectations": schema_expectations,
        "search_labels": {
            "recommended_queries": list(blueprint.get("search_terms", [])),
            "target_code": target_code,
            "accepted_code_family": _family(target_code),
        },
        "verification_checkpoints": _build_checkpoints(
            target_module=target_module,
            target_evidence_ids=target_evidence_ids,
            search_terms=list(blueprint.get("search_terms", [])),
            policy_schema_version=starting_schema_version,
            submit_schema_version=final_schema_version,
            target_code=target_code,
        ),
        "generator_metadata": {
            "generator_version": "v1",
            "source_bank": "icd10_mock.json",
            "seed": seed,
            "blueprint_slug": blueprint["slug"],
        },
    }


def generate_cases(
    *,
    seed: int = 7,
    bank_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Generate a deterministic set of synthetic V2 tasks."""

    repo_root = Path(__file__).resolve().parent
    bank_path = bank_path or (repo_root / "icd10_mock.json")
    code_index = _load_icd_index(bank_path)
    rng = random.Random(seed)
    plan = _case_plan_items()
    tasks: list[dict[str, Any]] = []
    usage_counter: dict[str, int] = {}
    shuffled_pools: dict[str, list[dict[str, Any]]] = {}

    for difficulty, blueprints in CASE_BLUEPRINTS.items():
        pool = deepcopy(blueprints)
        rng.shuffle(pool)
        shuffled_pools[difficulty] = pool

    for split, difficulty in plan:
        pool = shuffled_pools[difficulty]
        used = usage_counter.get(difficulty, 0)
        blueprint = deepcopy(pool[used % len(pool)])
        blueprint["difficulty"] = difficulty
        usage_counter[difficulty] = used + 1
        tasks.append(
            _build_task(
                blueprint=blueprint,
                split=split,
                sequence_index=len(tasks) + 1,
                rng=rng,
                code_index=code_index,
                seed=seed,
            )
        )

    return tasks


def write_generated_cases(
    *,
    output_path: Path,
    seed: int = 7,
    bank_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Generate tasks and write them to disk."""

    tasks = generate_cases(seed=seed, bank_path=bank_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(tasks, indent=2) + "\n", encoding="utf-8")
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic synthetic V2 tasks.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples") / "v2_tasks_synthetic.json",
        help="Path to the generated JSON file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Deterministic generation seed.",
    )
    parser.add_argument(
        "--bank",
        type=Path,
        default=Path("icd10_mock.json"),
        help="ICD candidate bank JSON file.",
    )
    args = parser.parse_args()
    write_generated_cases(output_path=args.output, seed=args.seed, bank_path=args.bank)


if __name__ == "__main__":
    main()
