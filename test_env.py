"""Quick smoke test for the Rveda environment."""

from models import MedicalAction, MedicalActionType
from server.rveda_environment import RvedaEnvironment


def main() -> None:
    env = RvedaEnvironment()

    initial_observation = env.reset()
    print("Initial observation:", initial_observation)

    search_observation = env.step(
        MedicalAction(action_type=MedicalActionType.SEARCH, query="diabetes")
    )
    print("Search observation:", search_observation)

    submit_observation = env.step(
        MedicalAction(action_type=MedicalActionType.SUBMIT, query="E11.40")
    )
    print("Final reward:", submit_observation.reward)


if __name__ == "__main__":
    main()
