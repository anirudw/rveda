"""Quick smoke test for the Rveda environment."""

import json

from starlette.testclient import TestClient

from rveda.models import MedicalAction, MedicalActionType
from rveda.server.app import app
from rveda.server.rveda_environment import RvedaEnvironment


SENSITIVE_REWARD_COMPONENT_KEYS = {
    "target_hit_bonus",
    "family_hit_bonus",
    "exact_match",
    "family_match",
    "family_bonus",
    "detail_relevant",
}


def main() -> None:
    env = RvedaEnvironment()

    initial_observation = env.reset(task_id="medium_endo_1")
    print("Initial observation:", initial_observation)

    search_observation = env.step(
        MedicalAction(action_type=MedicalActionType.SEARCH, query="diabetes")
    )
    print("Search observation:", search_observation)
    assert "target family" not in (search_observation.detailed_info or "")

    search_components = search_observation.grading.reward_components
    leaked_search_keys = SENSITIVE_REWARD_COMPONENT_KEYS.intersection(search_components.keys())
    assert not leaked_search_keys, f"Sensitive grading keys leaked in observation: {sorted(leaked_search_keys)}"

    submit_observation = env.step(
        MedicalAction(action_type=MedicalActionType.SUBMIT, query="E06.3")
    )
    print("Final reward:", submit_observation.reward)
    assert submit_observation.done is True
    assert submit_observation.last_error is None
    assert submit_observation.invalid_reason is None

    submit_components = submit_observation.grading.reward_components
    leaked_submit_keys = SENSITIVE_REWARD_COMPONENT_KEYS.intersection(submit_components.keys())
    assert not leaked_submit_keys, f"Sensitive grading keys leaked in submit grading: {sorted(leaked_submit_keys)}"

    terminal_misuse_observation = env.step(
        MedicalAction(action_type=MedicalActionType.SEARCH, query="diabetes")
    )
    assert terminal_misuse_observation.done is True
    assert terminal_misuse_observation.last_error == "terminal_misuse"
    assert terminal_misuse_observation.invalid_reason

    diagnostics_env = RvedaEnvironment()
    diagnostics_env.reset(task_id="easy_endo_1")
    unknown_code_observation = diagnostics_env.step(
        MedicalAction(action_type=MedicalActionType.DETAILS, query="ZZZ")
    )
    assert unknown_code_observation.last_error == "unknown_code"
    assert unknown_code_observation.invalid_reason == "Unknown code: ZZZ"

    unknown_submit_observation = diagnostics_env.step(
        MedicalAction(action_type=MedicalActionType.SUBMIT, query="ZZZ")
    )
    assert unknown_submit_observation.done is False
    assert unknown_submit_observation.last_error == "unknown_code"
    assert unknown_submit_observation.invalid_reason == "Unknown code: ZZZ"

    timeout_env = RvedaEnvironment()
    timeout_env.reset(task_id="easy_endo_1")
    timeout_observation = None
    for _ in range(8):
        timeout_observation = timeout_env.step(
            MedicalAction(action_type=MedicalActionType.SEARCH, query="diabetes")
        )
    assert timeout_observation is not None
    assert timeout_observation.done is True
    assert timeout_observation.last_error == "timeout"
    assert timeout_observation.invalid_reason == "Episode ended without a final SUBMIT decision."
    assert timeout_observation.metadata.get("timed_out") is True

    client = TestClient(app)
    reset_response = client.post("/reset", json={})
    assert reset_response.status_code == 200

    step_response = client.post(
        "/step",
        json={"action": {"action_type": "SEARCH", "query": "diabetes"}},
    )
    assert step_response.status_code == 200

    payload = step_response.json()
    info = payload.get("info", {})
    info_components = info.get("reward_components", {}) if isinstance(info, dict) else {}
    leaked_info_keys = SENSITIVE_REWARD_COMPONENT_KEYS.intersection(info_components.keys())
    assert not leaked_info_keys, f"Sensitive grading keys leaked in top-level info: {sorted(leaked_info_keys)}"

    # QUERY_EHR is stateful: verify through persistent websocket session.
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "reset", "data": {"task_id": "v2_easy_overweight_schema_v1"}}))
        _ = json.loads(ws.receive_text())

        ws.send_text(
            json.dumps(
                {
                    "type": "step",
                    "data": {
                        "action_type": "QUERY_EHR",
                        "module": "encounter_note",
                        "query": "BMI",
                    },
                }
            )
        )
        step_data = json.loads(ws.receive_text()).get("data", {})
        ws_obs = step_data.get("observation", {})
        ws_ehr_map = ws_obs.get("ehr_map", {})
        ws_revealed = ws_obs.get("revealed_evidence", [])

        encounter_state = ws_ehr_map.get("encounter_note", {})
        assert encounter_state.get("revealed_count") == 1
        assert encounter_state.get("query_budget_remaining") == 0
        assert ws_revealed and ws_revealed[0].get("evidence_id") == "ev_encounter_bmi_001"
        assert ws_obs.get("last_error") is None
        assert ws_obs.get("invalid_reason") is None

    print("Leakage smoke checks passed")


if __name__ == "__main__":
    main()
