import json

import numpy as np


class ObservationRenderer:
    def __init__(self, tokenizer, discrete_tokens=None, numerical_ranges=None,
                 bins=64, history_turns=8, chat_template=True):
        if int(bins) != bins or bins < 2 or int(history_turns) != history_turns or history_turns < 0:
            raise ValueError("Invalid renderer bin count or history length")
        self.tokenizer = tokenizer
        self.discrete_tokens = dict(discrete_tokens or {})
        self.numerical_ranges = dict(numerical_ranges or {})
        self.bins = int(bins)
        self.history_turns = int(history_turns)
        self.chat_template = bool(chat_template)
        for lower, upper in self.numerical_ranges.values():
            if not np.isfinite([lower, upper]).all() or upper <= lower:
                raise ValueError("Numerical feature ranges are finite and increasing")
        for mapping in self.discrete_tokens.values():
            for token in mapping.values():
                if token not in tokenizer.get_vocab():
                    raise ValueError("A reserved observation token is absent from the entry vocabulary")

    def features(self, value, path=""):
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, dict):
            return {key: self.features(item, path + "." + key if path else key)
                    for key, item in sorted(value.items())}
        if isinstance(value, (list, tuple)):
            return [self.features(item, path + "." + str(index)) for index, item in enumerate(value)]
        if path in self.discrete_tokens:
            key = str(value)
            if key not in self.discrete_tokens[path]:
                raise ValueError("Discrete observation value has no reserved token")
            return self.discrete_tokens[path][key]
        if path in self.numerical_ranges:
            lower, upper = self.numerical_ranges[path]
            if isinstance(value, bool) or not np.isfinite(value):
                raise ValueError("Continuous observation feature is nonfinite")
            fraction = np.clip((float(value) - lower) / (upper - lower), 0, 1)
            return int(np.rint((self.bins - 1) * fraction))
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError("Observation contains a nonfinite value")
        return value

    def encode(self, agent, environment, task, observation, legal_actions,
               history=(), role="", previous_action=None, elapsed_step=0):
        recent = list(history)[-self.history_turns:] if self.history_turns else []
        system = (f"You control participant {agent} in {environment}.\n"
                  "Use the task rules, your current observation, and the visible\n"
                  "interaction history to select an action.\n"
                  "Optimize the reward objective specified for your participant.\n"
                  "Choose an action from the current legal-action list.\n"
                  "Return the corresponding action identifier.")
        fields = self.features(observation)
        rendered = json.dumps(fields, ensure_ascii=False, sort_keys=True, allow_nan=False)
        history_text = json.dumps(recent, ensure_ascii=False, sort_keys=True, allow_nan=False)
        actions = "\n".join(f"{identifier}  {meaning}" for identifier, meaning in sorted(legal_actions.items()))
        decision = (f"TASK\n{task}\nPARTICIPANT\n{agent} {role}\n\n"
                    f"CURRENT OBSERVATION\n{rendered}\nPrevious action {previous_action}\n"
                    f"Elapsed episode step {elapsed_step}\n\nVISIBLE HISTORY\n{history_text}\n\n"
                    f"LEGAL ACTIONS\n{actions}\nDECISION\nReturn the identifier of your selected action.")
        if self.chat_template:
            identifiers = self.tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": decision}],
                tokenize=True, add_generation_prompt=True)
        else:
            identifiers = self.tokenizer.encode(system + "\n\n" + decision, add_special_tokens=True)
        return tuple(identifiers)
