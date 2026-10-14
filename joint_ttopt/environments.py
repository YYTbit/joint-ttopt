import importlib
import json
import selectors
import subprocess
from dataclasses import dataclass

from .storage import json_value


@dataclass(frozen=True)
class Frame:
    observations: dict
    legal_actions: dict
    rewards: dict
    terminated: bool
    task: str
    roles: dict
    histories: dict
    metrics: dict

    @classmethod
    def from_record(cls, value):
        return cls({int(agent): item for agent, item in value.get("observations", {}).items()},
                   {int(agent): {int(action): meaning for action, meaning in actions.items()}
                    for agent, actions in value.get("legal_actions", {}).items()},
                   {int(agent): float(reward) for agent, reward in value.get("rewards", {}).items()},
                   bool(value.get("terminated", False)), str(value.get("task", "")),
                   {int(agent): role for agent, role in value.get("roles", {}).items()},
                   {int(agent): history for agent, history in value.get("histories", {}).items()},
                   dict(value.get("metrics", {})))


class JsonEnvironment:
    def __init__(self, command, timeout=300.0):
        if not command or timeout <= 0:
            raise ValueError("Environment command and timeout are required")
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        text=True, bufsize=1)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.timeout = float(timeout)
        description = self.request("describe")
        self.agents = tuple(int(agent) for agent in description["agents"])
        self.name = str(description["name"])
        self.version = str(description["version"])
        self.data_kind = description.get("data_kind", "measured_run")

    def request(self, operation, **values):
        self.process.stdin.write(json.dumps(dict(values, operation=operation), allow_nan=False) + "\n")
        self.process.stdin.flush()
        if not self.selector.select(self.timeout):
            raise TimeoutError("Native benchmark service did not respond")
        response = self.process.stdout.readline()
        if not response:
            raise RuntimeError("Native benchmark service closed its response stream")
        value = json.loads(response)
        if "error" in value:
            raise RuntimeError(str(value["error"]))
        return value

    def reset(self, seed):
        return Frame.from_record(self.request("reset", seed=int(seed)))

    def step(self, actions):
        return Frame.from_record(self.request("step", actions={str(agent): action for agent, action in actions.items()}))

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.selector.close()


class ParallelEnvironment:
    def __init__(self, environment, action_catalog, name, task, roles=None, metric=None):
        self.environment = environment
        self.catalog = action_catalog
        self.name = name
        self.task = task
        self.roles = roles or {}
        self.metric = metric
        self.agent_names = tuple(environment.possible_agents)
        self.agents = tuple(range(len(self.agent_names)))
        self.version = str(getattr(environment, "metadata", {}).get("name", name))

    def frame(self, observations, rewards, terminated, infos):
        local, legal, history = {}, {}, {}
        for index, name in enumerate(self.agent_names):
            if name in observations:
                observation = observations[name]
                info = infos.get(name, {})
                if isinstance(observation, dict) and "observation" in observation:
                    mask = observation.get("action_mask")
                    observation = observation["observation"]
                else:
                    mask = info.get("action_mask")
                entries = self.catalog[name]
                legal[index] = {int(identifier): entry["meaning"] for identifier, entry in entries.items()
                                if mask is None or bool(mask[int(identifier)])}
                local[index] = observation
                history[index] = info.get("visible_history", [])
        metric_values = self.metric(infos, rewards) if terminated and self.metric else {}
        return Frame(local, legal, {index: float(rewards.get(name, 0.0))
                                   for index, name in enumerate(self.agent_names)},
                     terminated, self.task, self.roles, history, metric_values)

    def reset(self, seed):
        observations, infos = self.environment.reset(seed=seed)
        return self.frame(observations, {}, False, infos)

    def step(self, actions):
        native = {self.agent_names[agent]: self.catalog[self.agent_names[agent]][str(action)]["native"]
                  for agent, action in actions.items()}
        observations, rewards, terminated, truncated, infos = self.environment.step(native)
        done = not self.environment.agents or all(terminated.get(name, False) or truncated.get(name, False)
                                                 for name in self.agent_names)
        return self.frame(observations, rewards, done, infos)

    def close(self):
        self.environment.close()


def load_environment(factory, options):
    if ":" not in factory:
        raise ValueError("Environment factory uses module:function notation")
    module, name = factory.split(":", 1)
    constructor = getattr(importlib.import_module(module), name)
    environment = constructor(**options)
    for attribute in ["agents", "name", "version", "reset", "step", "close"]:
        if not hasattr(environment, attribute):
            raise ValueError("Environment adapter omitted a deployment interface field")
    return environment


def serve(environment, input_stream, output_stream):
    for line in input_stream:
        try:
            request = json.loads(line)
            operation = request["operation"]
            if operation == "describe":
                result = {"agents": list(environment.agents), "name": environment.name,
                          "version": environment.version,
                          "data_kind": getattr(environment, "data_kind", "measured_run")}
            elif operation == "reset":
                result = environment.reset(int(request["seed"])).__dict__
            elif operation == "step":
                result = environment.step({int(agent): int(action) for agent, action in request["actions"].items()}).__dict__
            else:
                raise ValueError("Unsupported environment operation")
            output_stream.write(json.dumps(json_value(result), allow_nan=False) + "\n")
        except Exception as error:
            output_stream.write(json.dumps({"error": str(error)}) + "\n")
        output_stream.flush()
