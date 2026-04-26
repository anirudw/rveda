"""Focused tests for Task 2.3 reward rubric behavior."""

from copy import deepcopy

from rveda.client import RvedaEnv
from rveda.models import MedicalAction, MedicalActionType
from rveda.server.rveda_environment import RvedaEnvironment


def _minimal_v2_task(env: RvedaEnvironment) -> dict:
    return next(
        task for task in env._v2_tasks if task["task_id"] == "v2_easy_overweight_schema_v1"
    )


def test_reward_metrics_are_exposed_on_submit() -> None:
    env = RvedaEnvironment()
    env.reset(task_id="easy_endo_1")

    observation = env.step(MedicalAction(action_type=MedicalActionType.SUBMIT, query="E66.3"))

    metrics = observation.reward_metrics

    assert metrics.terminal_correctness >= 0.0
    assert metrics.evidence_grounding >= 0.0
    assert metrics.schema_compliance >= 0.0
    assert metrics.format_validity >= 0.0
    assert metrics.process_discipline >= 0.0
    assert metrics.drift_adaptation >= 0.0
    assert metrics.final == observation.reward


def test_exact_submit_dominates_family_submit() -> None:
    env = RvedaEnvironment()
    env.reset(task_id="easy_endo_1")

    exact = env.step(MedicalAction(action_type=MedicalActionType.SUBMIT, query="E66.3")).reward

    env.reset(task_id="easy_endo_1")
    family = env.step(MedicalAction(action_type=MedicalActionType.SUBMIT, query="E66.01")).reward

    assert exact > family
    assert exact > 0.0
    assert family > 0.0


def test_process_reward_is_capped_for_repeated_searches() -> None:
    env = RvedaEnvironment()
    env.reset(task_id="easy_endo_1")

    first = env.step(MedicalAction(action_type=MedicalActionType.SEARCH, query="weight"))
    second = env.step(MedicalAction(action_type=MedicalActionType.SEARCH, query="weight"))

    assert first.reward <= 0.05
    assert second.reward <= 0.05
    assert second.reward <= first.reward
    assert second.reward_metrics.process_discipline <= first.reward_metrics.process_discipline


def test_client_parse_result_preserves_reward_metrics() -> None:
    client = RvedaEnv.__new__(RvedaEnv)

    result = client._parse_result(
        {
            "observation": {
                "patient_note": "",
                "search_results": [],
                "detailed_info": "",
                "current_reward": 0.72,
                "grading": {"action_type": "SUBMIT"},
                "reward_metrics": {
                    "terminal_correctness": 1.0,
                    "evidence_grounding": 0.75,
                    "schema_compliance": 1.0,
                    "format_validity": 1.0,
                    "process_discipline": 0.8,
                    "drift_adaptation": 1.0,
                    "final": 0.72,
                },
                "metadata": {"step": 8},
            },
            "reward": 0.72,
            "done": True,
        }
    )

    assert result.observation.reward_metrics.final == 0.72
    assert result.observation.reward_metrics.terminal_correctness == 1.0
    assert result.observation.reward_metrics.process_discipline == 0.8


def test_drift_adaptation_requires_post_drift_schema_validation() -> None:
    env = RvedaEnvironment()
    drift_task = deepcopy(_minimal_v2_task(env))
    drift_task["task_id"] = "v2_drift_reward_missing_validation"
    drift_task["drift"] = {
        "enabled": True,
        "trigger_step": 1,
        "from_schema_version": "v1",
        "to_schema_version": "v2",
        "required_fields": [
            "diagnosis_code",
            "schema_version",
            "evidence_ids",
            "policy_attestations",
        ],
    }
    env._v2_example_tasks.append(drift_task)

    env.reset(task_id="v2_drift_reward_missing_validation")
    env.step(MedicalAction(action_type=MedicalActionType.QUERY_EHR, module="encounter_note", query="BMI"))
    env.step(MedicalAction(action_type=MedicalActionType.CHECK_POLICY))
    env.step(
        MedicalAction(
            action_type=MedicalActionType.REASONING_LOG,
            payload={
                "candidate_code": "E66.3",
                "rationale": "The encounter note documents BMI 27 and weight-management counseling.",
                "evidence_ids": ["ev_encounter_bmi_001"],
                "policy_rule_ids": [],
            },
        )
    )
    observation = env.step(MedicalAction(action_type=MedicalActionType.SUBMIT, query="E66.3"))

    assert observation.done is True
    assert observation.reward_metrics.drift_adaptation == 0.0


def test_drift_adaptation_rewards_validated_post_drift_submit() -> None:
    env = RvedaEnvironment()
    drift_task = deepcopy(_minimal_v2_task(env))
    drift_task["task_id"] = "v2_drift_reward_validated_submit"
    drift_task["drift"] = {
        "enabled": True,
        "trigger_step": 1,
        "from_schema_version": "v1",
        "to_schema_version": "v2",
        "required_fields": [
            "diagnosis_code",
            "schema_version",
            "evidence_ids",
            "policy_attestations",
        ],
    }
    env._v2_example_tasks.append(drift_task)

    env.reset(task_id="v2_drift_reward_validated_submit")
    env.step(MedicalAction(action_type=MedicalActionType.QUERY_EHR, module="encounter_note", query="BMI"))
    env.step(MedicalAction(action_type=MedicalActionType.CHECK_POLICY))
    validation = env.step(
        MedicalAction(
            action_type=MedicalActionType.VALIDATE_CLAIM_SCHEMA,
            payload={
                "diagnosis_code": "E66.3",
                "schema_version": "v2",
                "evidence_ids": ["ev_encounter_bmi_001"],
                "policy_attestations": ["policy_v2_attestation_required"],
            },
        )
    )
    env.step(
        MedicalAction(
            action_type=MedicalActionType.REASONING_LOG,
            payload={
                "candidate_code": "E66.3",
                "rationale": "The encounter note documents BMI 27 and weight-management counseling.",
                "evidence_ids": ["ev_encounter_bmi_001"],
                "policy_rule_ids": [],
            },
        )
    )
    observation = env.step(MedicalAction(action_type=MedicalActionType.SUBMIT, query="E66.3"))

    assert validation.reward_metrics.drift_adaptation == 0.75
    assert observation.done is True
    assert observation.reward_metrics.drift_adaptation == 1.0
