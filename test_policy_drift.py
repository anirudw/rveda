"""Focused tests for Task 2.1 policy and schema-drift behavior."""

from copy import deepcopy

from rveda.client import RvedaEnv
from rveda.inference import parse_action_or_fallback
from rveda.models import ClaimDraftPayload, MedicalAction, MedicalActionType, ReasoningLogPayload
from rveda.server.policy_engine import PolicyEngine
from rveda.server.rveda_environment import RvedaEnvironment


def _minimal_v2_task(env: RvedaEnvironment) -> dict:
    return next(
        task for task in env._v2_tasks if task["task_id"] == "v2_easy_overweight_schema_v1"
    )


def test_policy_engine_reset_and_check_policy() -> None:
    env = RvedaEnvironment()
    engine = PolicyEngine()
    engine.reset(_minimal_v2_task(env))

    assert engine.policy_state().checked is False

    result = engine.check_policy()
    policy_state = engine.policy_state()

    assert result.last_error is None
    assert policy_state.checked is True
    assert policy_state.policy_version == "payer_policy_v1"
    assert policy_state.active_schema_version == "v1"


def test_policy_engine_drift_and_validation() -> None:
    env = RvedaEnvironment()
    drift_task = deepcopy(_minimal_v2_task(env))
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

    engine = PolicyEngine()
    engine.reset(drift_task)
    engine.maybe_trigger_drift(1)
    result = engine.validate_claim(
        ClaimDraftPayload(
            diagnosis_code="E66.3",
            schema_version="v2",
            evidence_ids=["ev_encounter_bmi_001"],
            policy_attestations=["policy_v2_attestation_required"],
        ),
        revealed_evidence={},
    )

    assert engine.drift_notice is not None
    assert engine.active_schema_version == "v2"
    engine.check_policy()
    assert engine.policy_state().active_schema_version == "v2"
    assert result.last_error == "unknown_evidence_ids"


def test_check_policy_reveals_active_schema_state() -> None:
    env = RvedaEnvironment()
    initial_observation = env.reset(task_id="v2_easy_overweight_schema_v1")

    assert initial_observation.policy_state.checked is False
    assert initial_observation.policy_state.active_schema_version == ""
    assert initial_observation.drift_notice is None

    observation = env.step(MedicalAction(action_type=MedicalActionType.CHECK_POLICY))

    assert observation.last_error is None
    assert observation.policy_state.checked is True
    assert observation.policy_state.policy_version == "payer_policy_v1"
    assert observation.policy_state.active_schema_version == "v1"
    assert observation.policy_state.claim_schema.version == "v1"
    assert observation.policy_state.claim_schema.required_fields == [
        "diagnosis_code",
        "schema_version",
        "evidence_ids",
        "reasoning_log_id",
    ]


def test_validate_claim_schema_accepts_valid_revealed_claim() -> None:
    env = RvedaEnvironment()
    env.reset(task_id="v2_easy_overweight_schema_v1")
    env.step(
        MedicalAction(
            action_type=MedicalActionType.QUERY_EHR,
            module="encounter_note",
            query="BMI",
        )
    )

    observation = env.step(
        MedicalAction(
            action_type=MedicalActionType.VALIDATE_CLAIM_SCHEMA,
            payload={
                "diagnosis_code": "E66.3",
                "schema_version": "v1",
                "evidence_ids": ["ev_encounter_bmi_001"],
                "policy_attestations": [],
                "reasoning_log_id": "draft_reasoning_log_001",
            },
        )
    )

    assert observation.last_error is None
    assert observation.invalid_reason is None
    assert "accepted schema v1" in observation.detailed_info
    assert observation.reward > 0


def test_validate_claim_schema_requires_missing_fields() -> None:
    env = RvedaEnvironment()
    env.reset(task_id="v2_easy_overweight_schema_v1")
    env.step(
        MedicalAction(
            action_type=MedicalActionType.QUERY_EHR,
            module="encounter_note",
            query="BMI",
        )
    )

    observation = env.step(
        MedicalAction(
            action_type=MedicalActionType.VALIDATE_CLAIM_SCHEMA,
            payload={
                "diagnosis_code": "E66.3",
                "schema_version": "v1",
                "evidence_ids": ["ev_encounter_bmi_001"],
                "policy_attestations": [],
            },
        )
    )

    assert observation.last_error == "schema_validation_failed"
    assert "reasoning_log_id" in observation.invalid_reason


def test_validate_claim_schema_rejects_wrong_schema_version() -> None:
    env = RvedaEnvironment()
    env.reset(task_id="v2_easy_overweight_schema_v1")
    env.step(
        MedicalAction(
            action_type=MedicalActionType.QUERY_EHR,
            module="encounter_note",
            query="BMI",
        )
    )

    observation = env.step(
        MedicalAction(
            action_type=MedicalActionType.VALIDATE_CLAIM_SCHEMA,
            payload={
                "diagnosis_code": "E66.3",
                "schema_version": "v9",
                "evidence_ids": ["ev_encounter_bmi_001"],
                "policy_attestations": [],
                "reasoning_log_id": "draft_reasoning_log_001",
            },
        )
    )

    assert observation.last_error == "schema_version_mismatch"
    assert "expected v1" in observation.invalid_reason


def test_verify_reasoning_log_accepts_grounded_revealed_evidence() -> None:
    env = RvedaEnvironment()
    env.reset(task_id="v2_easy_overweight_schema_v1")
    env.step(
        MedicalAction(
            action_type=MedicalActionType.QUERY_EHR,
            module="encounter_note",
            query="BMI",
        )
    )

    verified, reasoning_log, last_error, invalid_reason = env.verify_reasoning_log(
        ReasoningLogPayload(
            candidate_code="E66.3",
            rationale="The encounter note documents BMI 27 and weight-management counseling.",
            evidence_ids=["ev_encounter_bmi_001"],
            policy_rule_ids=[],
        )
    )

    assert verified is True
    assert reasoning_log is not None
    assert reasoning_log.verified is True
    assert reasoning_log.candidate_code == "E66.3"
    assert reasoning_log.evidence_ids == ["ev_encounter_bmi_001"]
    assert last_error is None
    assert invalid_reason is None


def test_verify_reasoning_log_rejects_unrevealed_evidence() -> None:
    env = RvedaEnvironment()
    env.reset(task_id="v2_easy_overweight_schema_v1")

    verified, reasoning_log, last_error, invalid_reason = env.verify_reasoning_log(
        ReasoningLogPayload(
            candidate_code="E66.3",
            rationale="The encounter note supports overweight coding.",
            evidence_ids=["ev_encounter_bmi_001"],
            policy_rule_ids=[],
        )
    )

    assert verified is False
    assert reasoning_log is None
    assert last_error == "unknown_evidence_ids"
    assert "not been revealed" in invalid_reason


def test_verify_reasoning_log_requires_rationale() -> None:
    env = RvedaEnvironment()
    env.reset(task_id="v2_easy_overweight_schema_v1")
    env.step(
        MedicalAction(
            action_type=MedicalActionType.QUERY_EHR,
            module="encounter_note",
            query="BMI",
        )
    )

    verified, reasoning_log, last_error, invalid_reason = env.verify_reasoning_log(
        ReasoningLogPayload(
            candidate_code="E66.3",
            rationale="",
            evidence_ids=["ev_encounter_bmi_001"],
            policy_rule_ids=[],
        )
    )

    assert verified is False
    assert reasoning_log is None
    assert last_error == "missing_rationale"
    assert "requires a rationale" in invalid_reason


def test_reasoning_log_action_stores_verified_record() -> None:
    env = RvedaEnvironment()
    env.reset(task_id="v2_easy_overweight_schema_v1")
    env.step(
        MedicalAction(
            action_type=MedicalActionType.QUERY_EHR,
            module="encounter_note",
            query="BMI",
        )
    )

    observation = env.step(
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

    assert observation.last_error is None
    assert observation.invalid_reason is None
    assert observation.reasoning_log is not None
    assert observation.reasoning_log.verified is True
    assert observation.reasoning_log_verified is True
    assert observation.reasoning_log.candidate_code == "E66.3"
    assert "REASONING_LOG accepted" in observation.detailed_info
    assert observation.reward > 0


def test_reasoning_log_action_preserves_failed_attempt() -> None:
    env = RvedaEnvironment()
    env.reset(task_id="v2_easy_overweight_schema_v1")

    observation = env.step(
        MedicalAction(
            action_type=MedicalActionType.REASONING_LOG,
            payload={
                "candidate_code": "E66.3",
                "rationale": "The encounter note supports overweight coding.",
                "evidence_ids": ["ev_encounter_bmi_001"],
                "policy_rule_ids": [],
            },
        )
    )

    assert observation.last_error == "unknown_evidence_ids"
    assert observation.reasoning_log is not None
    assert observation.reasoning_log.verified is False
    assert observation.reasoning_log_verified is False
    assert observation.reasoning_log.evidence_ids == ["ev_encounter_bmi_001"]


def test_submit_requires_verified_reasoning_log_for_v2_task() -> None:
    env = RvedaEnvironment()
    env.reset(task_id="v2_easy_overweight_schema_v1")

    observation = env.step(
        MedicalAction(action_type=MedicalActionType.SUBMIT, query="E66.3")
    )

    assert observation.done is False
    assert observation.reward == 0.0
    assert observation.last_error == "missing_reasoning_log"
    assert "verified REASONING_LOG" in observation.invalid_reason


def test_submit_succeeds_after_verified_reasoning_log() -> None:
    env = RvedaEnvironment()
    env.reset(task_id="v2_easy_overweight_schema_v1")
    env.step(
        MedicalAction(
            action_type=MedicalActionType.QUERY_EHR,
            module="encounter_note",
            query="BMI",
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

    observation = env.step(
        MedicalAction(action_type=MedicalActionType.SUBMIT, query="E66.3")
    )

    assert observation.done is True
    assert observation.last_error is None
    assert observation.invalid_reason is None
    assert observation.reward > 0


def test_submit_rejects_reasoning_log_candidate_mismatch() -> None:
    env = RvedaEnvironment()
    env.reset(task_id="v2_easy_overweight_schema_v1")
    env.step(
        MedicalAction(
            action_type=MedicalActionType.QUERY_EHR,
            module="encounter_note",
            query="BMI",
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

    observation = env.step(
        MedicalAction(action_type=MedicalActionType.SUBMIT, query="E11.9")
    )

    assert observation.done is False
    assert observation.reward == 0.0
    assert observation.last_error == "reasoning_log_candidate_mismatch"
    assert "does not match submitted code" in observation.invalid_reason


def test_drift_notice_surfaces_when_schema_changes_mid_episode() -> None:
    env = RvedaEnvironment()
    drift_task = deepcopy(_minimal_v2_task(env))
    drift_task["task_id"] = "v2_policy_drift_test"
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

    env.reset(task_id="v2_policy_drift_test")
    observation = env.step(MedicalAction(action_type=MedicalActionType.CHECK_POLICY))

    assert observation.drift_notice is not None
    assert observation.drift_notice.trigger_step == 1
    assert observation.drift_notice.from_schema_version == "v1"
    assert observation.drift_notice.to_schema_version == "v2"
    assert observation.policy_state.active_schema_version == "v2"
    assert observation.policy_state.claim_schema.required_fields == [
        "diagnosis_code",
        "schema_version",
        "evidence_ids",
        "policy_attestations",
    ]


def test_client_step_payload_includes_validation_payload() -> None:
    client = RvedaEnv.__new__(RvedaEnv)

    payload = client._step_payload(
        MedicalAction(
            action_type=MedicalActionType.VALIDATE_CLAIM_SCHEMA,
            payload={
                "diagnosis_code": "E66.3",
                "schema_version": "v1",
                "evidence_ids": ["ev_encounter_bmi_001"],
                "policy_attestations": [],
                "reasoning_log_id": "draft_reasoning_log_001",
            },
        )
    )

    assert payload == {
        "action_type": "VALIDATE_CLAIM_SCHEMA",
        "query": "",
        "payload": {
            "diagnosis_code": "E66.3",
            "schema_version": "v1",
            "evidence_ids": ["ev_encounter_bmi_001"],
            "policy_attestations": [],
            "reasoning_log_id": "draft_reasoning_log_001",
        },
    }


def test_client_step_payload_includes_reasoning_log_payload() -> None:
    client = RvedaEnv.__new__(RvedaEnv)

    payload = client._step_payload(
        MedicalAction(
            action_type=MedicalActionType.REASONING_LOG,
            payload={
                "candidate_code": "E66.3",
                "rationale": "The encounter note documents BMI 27 and counseling.",
                "evidence_ids": ["ev_encounter_bmi_001"],
                "policy_rule_ids": [],
            },
        )
    )

    assert payload == {
        "action_type": "REASONING_LOG",
        "query": "",
        "payload": {
            "candidate_code": "E66.3",
            "rationale": "The encounter note documents BMI 27 and counseling.",
            "evidence_ids": ["ev_encounter_bmi_001"],
            "policy_rule_ids": [],
        },
    }


def test_client_parse_result_preserves_policy_and_drift_fields() -> None:
    client = RvedaEnv.__new__(RvedaEnv)

    result = client._parse_result(
        {
            "observation": {
                "patient_note": "",
                "search_results": [],
                "detailed_info": "CHECK_POLICY revealed payer_policy_v2 with schema v2 and 1 rule(s).",
                "current_reward": 0.02,
                "grading": {"action_type": "CHECK_POLICY"},
                "policy_state": {
                    "checked": True,
                    "policy_version": "payer_policy_v2",
                    "active_schema_version": "v2",
                    "rules": [
                        {
                            "rule_id": "policy_v2_attestation_required",
                            "description": "Claims must include a policy attestation after drift.",
                        }
                    ],
                    "claim_schema": {
                        "version": "v2",
                        "required_fields": [
                            "diagnosis_code",
                            "schema_version",
                            "evidence_ids",
                            "policy_attestations",
                        ],
                    },
                },
                "drift_notice": {
                    "active": True,
                    "trigger_step": 1,
                    "from_schema_version": "v1",
                    "to_schema_version": "v2",
                    "message": "Policy/schema drift activated at step 1: v1 -> v2",
                },
                "metadata": {"step": 1},
            },
            "reward": 0.02,
            "done": False,
        }
    )

    assert result.observation.policy_state.checked is True
    assert result.observation.policy_state.active_schema_version == "v2"
    assert result.observation.policy_state.claim_schema.required_fields == [
        "diagnosis_code",
        "schema_version",
        "evidence_ids",
        "policy_attestations",
    ]
    assert result.observation.drift_notice is not None
    assert result.observation.drift_notice.to_schema_version == "v2"


def test_client_parse_result_preserves_reasoning_log_fields() -> None:
    client = RvedaEnv.__new__(RvedaEnv)

    result = client._parse_result(
        {
            "observation": {
                "patient_note": "",
                "search_results": [],
                "detailed_info": "REASONING_LOG accepted rl_episode_3 for candidate code E66.3.",
                "current_reward": 0.04,
                "grading": {"action_type": "REASONING_LOG"},
                "reasoning_log": {
                    "reasoning_log_id": "rl_episode_3",
                    "candidate_code": "E66.3",
                    "rationale": "The encounter note documents BMI 27 and counseling.",
                    "evidence_ids": ["ev_encounter_bmi_001"],
                    "policy_rule_ids": [],
                    "verified": True,
                },
                "reasoning_log_verified": True,
                "metadata": {"step": 3},
            },
            "reward": 0.04,
            "done": False,
        }
    )

    assert result.observation.reasoning_log is not None
    assert result.observation.reasoning_log.reasoning_log_id == "rl_episode_3"
    assert result.observation.reasoning_log.candidate_code == "E66.3"
    assert result.observation.reasoning_log_verified is True


def test_inference_parser_accepts_check_policy() -> None:
    parsed = parse_action_or_fallback('{"action_type":"CHECK_POLICY"}')

    assert parsed == {
        "action_type": "CHECK_POLICY",
        "query": "",
        "module": None,
        "payload": None,
    }


def test_inference_parser_accepts_schema_validation_payload() -> None:
    parsed = parse_action_or_fallback(
        '{"action_type":"VALIDATE_CLAIM_SCHEMA","payload":{"diagnosis_code":"E66.3","schema_version":"v1","evidence_ids":["ev_encounter_bmi_001"],"policy_attestations":[],"reasoning_log_id":"draft_reasoning_log_001"}}'
    )

    assert parsed == {
        "action_type": "VALIDATE_CLAIM_SCHEMA",
        "query": "",
        "module": None,
        "payload": {
            "diagnosis_code": "E66.3",
            "schema_version": "v1",
            "evidence_ids": ["ev_encounter_bmi_001"],
            "policy_attestations": [],
            "reasoning_log_id": "draft_reasoning_log_001",
        },
    }


def test_inference_parser_accepts_reasoning_log_payload() -> None:
    parsed = parse_action_or_fallback(
        '{"action_type":"REASONING_LOG","payload":{"candidate_code":"E66.3","rationale":"The encounter note documents BMI 27 and counseling.","evidence_ids":["ev_encounter_bmi_001"],"policy_rule_ids":[]}}'
    )

    assert parsed == {
        "action_type": "REASONING_LOG",
        "query": "",
        "module": None,
        "payload": {
            "candidate_code": "E66.3",
            "rationale": "The encounter note documents BMI 27 and counseling.",
            "evidence_ids": ["ev_encounter_bmi_001"],
            "policy_rule_ids": [],
        },
    }
