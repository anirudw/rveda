"""Smoke tests for the V2 EHR Fog-of-War slice."""

from client import RvedaEnv
from models import MedicalAction, MedicalActionType
from server.rveda_environment import RvedaEnvironment


def test_query_ehr_reveals_hidden_evidence() -> None:
    env = RvedaEnvironment()
    initial_observation = env.reset(task_id="v2_easy_overweight_schema_v1")

    assert initial_observation.patient_note == ""
    assert initial_observation.ehr_map["encounter_note"].status == "closed"
    assert initial_observation.revealed_evidence == []

    observation = env.step(
        MedicalAction(
            action_type=MedicalActionType.QUERY_EHR,
            module="encounter_note",
            query="BMI",
        )
    )

    assert observation.last_error is None
    assert observation.invalid_reason is None
    assert observation.ehr_map["encounter_note"].status == "exhausted"
    assert observation.ehr_map["encounter_note"].query_budget_remaining == 0
    assert [evidence.evidence_id for evidence in observation.revealed_evidence] == [
        "ev_encounter_bmi_001"
    ]
    assert observation.reward > 0


def test_query_ehr_invalid_module_sets_error() -> None:
    env = RvedaEnvironment()
    env.reset(task_id="v2_easy_overweight_schema_v1")

    observation = env.step(
        MedicalAction(
            action_type=MedicalActionType.QUERY_EHR,
            module="missing_module",
            query="BMI",
        )
    )

    assert observation.last_error == "invalid_module"
    assert "missing_module" in observation.invalid_reason


def test_query_ehr_budget_exhaustion_sets_error() -> None:
    env = RvedaEnvironment()
    env.reset(task_id="v2_easy_overweight_schema_v1")

    action = MedicalAction(
        action_type=MedicalActionType.QUERY_EHR,
        module="encounter_note",
        query="BMI",
    )
    env.step(action)
    observation = env.step(action)

    assert observation.last_error == "query_budget_exhausted"
    assert "exhausted" in observation.invalid_reason


def test_client_step_payload_includes_query_ehr_module() -> None:
    client = RvedaEnv.__new__(RvedaEnv)

    payload = client._step_payload(
        MedicalAction(
            action_type=MedicalActionType.QUERY_EHR,
            module="encounter_note",
            query="BMI",
        )
    )

    assert payload == {
        "action_type": "QUERY_EHR",
        "module": "encounter_note",
        "query": "BMI",
    }


def test_client_parse_result_preserves_ehr_fields() -> None:
    client = RvedaEnv.__new__(RvedaEnv)

    result = client._parse_result(
        {
            "observation": {
                "patient_note": "",
                "search_results": [],
                "detailed_info": "QUERY_EHR revealed 1 evidence snippet(s) from encounter_note.",
                "current_reward": 0.03,
                "grading": {"action_type": "QUERY_EHR"},
                "ehr_map": {
                    "encounter_note": {
                        "status": "exhausted",
                        "query_budget_remaining": 0,
                        "revealed_count": 1,
                    }
                },
                "revealed_evidence": [
                    {
                        "evidence_id": "ev_encounter_bmi_001",
                        "module": "encounter_note",
                        "text": "Routine checkup documents BMI 27 with diet and exercise counseling for weight management.",
                        "supports_codes": ["E66.3"],
                        "metadata": {},
                    }
                ],
                "last_error": "query_budget_exhausted",
                "invalid_reason": "EHR module query budget exhausted: encounter_note",
                "metadata": {"step": 2},
            },
            "reward": 0.0,
            "done": False,
        }
    )

    assert result.observation.ehr_map["encounter_note"].status == "exhausted"
    assert result.observation.ehr_map["encounter_note"].query_budget_remaining == 0
    assert [item.evidence_id for item in result.observation.revealed_evidence] == [
        "ev_encounter_bmi_001"
    ]
    assert result.observation.last_error == "query_budget_exhausted"
    assert result.observation.invalid_reason == (
        "EHR module query budget exhausted: encounter_note"
    )
