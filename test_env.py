"""Quick smoke test for the Rveda environment."""

from starlette.testclient import TestClient

from models import MedicalAction, MedicalActionType
from server.app import app
from server.rveda_environment import RvedaEnvironment


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

    initial_observation = env.reset()
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
        MedicalAction(action_type=MedicalActionType.SUBMIT, query="E11.40")
    )
    print("Final reward:", submit_observation.reward)

    submit_components = submit_observation.grading.reward_components
    leaked_submit_keys = SENSITIVE_REWARD_COMPONENT_KEYS.intersection(submit_components.keys())
    assert not leaked_submit_keys, f"Sensitive grading keys leaked in submit grading: {sorted(leaked_submit_keys)}"

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

    print("Leakage smoke checks passed")


if __name__ == "__main__":
    main()
