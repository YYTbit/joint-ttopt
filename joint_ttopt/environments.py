import importlib
import json
import math
import os
import selectors
import subprocess
import time
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
    def __init__(self, command, timeout=300.0, env=None):
        if not command or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Environment command and timeout are required")
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
        os.set_blocking(self.process.stdin.fileno(), False)
        os.set_blocking(self.process.stdout.fileno(), False)
        self.buffer = bytearray()
        self.closed = False
        self.timeout = float(timeout)
        try:
            description = self.request("describe")
            agents = description["agents"]
            if (not agents or any(type(agent) is not int or agent < 0 for agent in agents)
                    or len(set(agents)) != len(agents) or not description["name"] or not description["version"]):
                raise ValueError("Native benchmark description requires distinct participants and a version")
            count = description.get("action_count")
            if count is not None and (type(count) is not int or count < 1):
                raise ValueError("The native action vocabulary has a positive integer size")
        except BaseException:
            self.close(force=True)
            raise
        self.agents = tuple(int(agent) for agent in description["agents"])
        self.name = str(description["name"])
        self.version = str(description["version"])
        self.data_kind = description.get("data_kind", "measured_run")
        self.action_count = description.get("action_count")

    def request(self, operation, **values):
        if self.closed:
            raise RuntimeError("Native benchmark service is closed")
        payload = (json.dumps(dict(values, operation=operation), allow_nan=False) + "\n").encode()
        deadline = time.monotonic() + self.timeout
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(self.process.stdin, selectors.EVENT_WRITE)
                offset = 0
                while offset < len(payload):
                    self.wait(selector, deadline)
                    try:
                        offset += os.write(self.process.stdin.fileno(), payload[offset:])
                    except BlockingIOError:
                        continue
                selector.unregister(self.process.stdin)
                selector.register(self.process.stdout, selectors.EVENT_READ)
                while b"\n" not in self.buffer:
                    self.wait(selector, deadline)
                    try:
                        chunk = os.read(self.process.stdout.fileno(), 65536)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        raise RuntimeError("Native benchmark service closed its response stream")
                    self.buffer.extend(chunk)
            response, _, remainder = self.buffer.partition(b"\n")
            self.buffer = bytearray(remainder)
            value = json.loads(response)
            if not isinstance(value, dict):
                raise ValueError("Native benchmark service returns a JSON object")
            if "error" in value:
                raise RuntimeError(str(value["error"]))
            return value
        except BaseException:
            self.close(force=True)
            raise

    @staticmethod
    def wait(selector, deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not selector.select(remaining):
            raise TimeoutError("Native benchmark service exceeded the request deadline")

    def reset(self, seed):
        return Frame.from_record(self.request("reset", seed=int(seed)))

    def step(self, actions):
        return Frame.from_record(self.request("step", actions={str(agent): action for agent, action in actions.items()}))

    def close(self, force=False):
        if self.closed:
            return
        self.closed = True
        if self.process.poll() is None:
            # EOF lets native workers release their SDK resources before termination is needed.
            self.process.stdin.close()
            if force:
                self.process.kill()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        self.process.stdin.close()
        self.process.stdout.close()


class ParallelEnvironment:
    def __init__(self, environment, action_catalog, name, task, roles=None, metric=None):
        self.environment = environment
        self.catalog = {name: {int(identifier): entry for identifier, entry in entries.items()}
                        for name, entries in action_catalog.items()}
        if any(len(self.catalog[name]) != len(entries) for name, entries in action_catalog.items()):
            raise ValueError("Native action identifiers are distinct after integer conversion")
        self.name = name
        self.task = task
        self.roles = roles or {}
        self.metric = metric
        self.agent_names = tuple(environment.possible_agents)
        if set(self.catalog) != set(self.agent_names) or any(not entries for entries in self.catalog.values()):
            raise ValueError("Each native participant has a nonempty action catalog")
        self.action_count = 1 + max(identifier for entries in self.catalog.values() for identifier in entries)
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
        native = {self.agent_names[agent]: self.catalog[self.agent_names[agent]][int(action)]["native"]
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
    try:
        for line in input_stream:
            try:
                request = json.loads(line)
                operation = request["operation"]
                if operation == "describe":
                    result = {"agents": list(environment.agents), "name": environment.name,
                              "version": environment.version,
                              "action_count": getattr(environment, "action_count", None),
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
    finally:
        environment.close()
