"""Executable coordination example for the shared environment protocol."""

import sys

from joint_ttopt.environments import Frame, serve


class ExampleEnvironment:
    agents = (0, 1)
    name = "coordination_example"
    version = "example-v1"
    data_kind = "software_fixture"

    def reset(self, seed):
        actions = {agent: {0: "Choose zero", 1: "Choose one"} for agent in self.agents}
        return Frame({agent: {"round": 0} for agent in self.agents}, actions,
                     dict.fromkeys(self.agents, 0.0), False,
                     "Choose the same action as the other participant.", {}, {}, {})

    def step(self, actions):
        if set(actions) != set(self.agents) or any(value not in (0, 1) for value in actions.values()):
            raise ValueError("The example requires one legal action per participant")
        reward = float(actions[0] == actions[1])
        return Frame({}, {}, dict.fromkeys(self.agents, reward), True,
                     "Coordination complete.", {}, {}, {"coordination": reward})

    def close(self):
        pass


if __name__ == "__main__":
    serve(ExampleEnvironment(), sys.stdin, sys.stdout)
