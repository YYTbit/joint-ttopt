import contextlib
import importlib
import importlib.util
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from .environments import Frame


def source_revision(source, expected=None):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if expected is not None and revision != expected:
        raise ValueError("Native benchmark source revision differs from the configured release")
    return revision


class AvalonEnvironment:
    def __init__(self, source="external/AvalonBench", expected_revision=None,
                 players=5, focal_player=0, merlin=True, focal_role=None):
        source = Path(source)
        self.version = source_revision(source, expected_revision)
        if players != 5 or not 0 <= focal_player < players:
            raise ValueError("The paper's Avalon interface uses five players and a valid focal player")
        sys.path.insert(0, str(source.resolve()))
        module = importlib.import_module("strategist.Avalon.engine")
        self.configuration = module.AvalonBasicConfig.from_num_players(players, merlin=merlin)
        with contextlib.redirect_stdout(sys.stderr):
            self.engine = module.AvalonGameEnvironment(self.configuration)
        self.agents = tuple(range(players))
        self.name = "AvalonBench"
        self.focal = focal_player
        if focal_role not in (None, "Merlin", "Assassin", "Servant"):
            raise ValueError("The focal Avalon role is Merlin, Assassin, Servant, or role-agnostic")
        self.focal_role = focal_role
        self.teams = list(itertools.combinations(self.agents, 2)) + list(itertools.combinations(self.agents, 3))
        self.team_catalog = {index: frozenset(team) for index, team in enumerate(self.teams)}
        self.vote_approve, self.vote_reject = 20, 21
        self.quest_success, self.quest_failure = 22, 23
        self.assassination_start = 24
        self.history = []
        self.task = ("Play five-player Avalon using the released quest and role rules. "
                     "Loyal participants seek a loyal victory; evil participants seek an evil victory.")

    def reset(self, seed):
        np.random.seed(seed)
        with contextlib.redirect_stdout(sys.stderr):
            self.engine.reset()
        if self.focal_role is not None:
            candidates = [agent for agent in self.agents if self.engine.get_role(agent)[1] == self.focal_role]
            if not candidates:
                raise ValueError("The configured Avalon game omits the requested focal role")
            target = int(np.random.choice(candidates))
            self.engine.roles[[self.focal, target]] = self.engine.roles[[target, self.focal]]
            self.engine.is_good[[self.focal, target]] = self.engine.is_good[[target, self.focal]]
        self.history = []
        return self.frame()

    def frame(self, rewards=None):
        game = self.engine
        if game.done:
            outcomes = {agent: float(bool(game.is_good[agent]) == bool(game.good_victory)) for agent in self.agents}
            return Frame({}, {}, outcomes, True, self.task, {}, {},
                         {"win_rate": 100 * outcomes[self.focal],
                          "focal_role": self.engine.get_role(self.focal)[1]})
        phase = int(game.phase)
        if phase == 0:
            actors = [int(game.quest_leader)]
            actions = {identifier: "Propose quest team " + ", ".join(map(str, sorted(team)))
                       for identifier, team in self.team_catalog.items() if len(team) == game.get_team_size()}
            legal = {actors[0]: actions}
        elif phase == 1:
            actors = list(self.agents)
            legal = {agent: {self.vote_approve: "Approve the proposed team", self.vote_reject: "Reject the proposed team"}
                     for agent in actors}
        elif phase == 2:
            actors = sorted(game.get_current_quest_team())
            legal = {agent: {self.quest_success: "Support the quest"} for agent in actors}
            for agent in actors:
                if not game.is_good[agent]:
                    legal[agent][self.quest_failure] = "Sabotage the quest"
        else:
            actors = [int(game.get_assassin())]
            legal = {actors[0]: {self.assassination_start + target: "Identify participant " + str(target) + " as Merlin"
                                  for target in self.agents if target != actors[0]}}
        observations, roles = {}, {}
        for agent in actors:
            role, name, alignment = game.get_role(agent)
            observations[agent] = {"phase": game.get_phase()[1], "quest": int(game.turn),
                                   "leader": int(game.quest_leader), "team": sorted(game.quest_team),
                                   "quest_results": [bool(value) for value in game.quest_results],
                                   "known_alignments": [int(value) for value in game.get_partial_sides(agent)]}
            roles[agent] = name
        return Frame(observations, legal, rewards or dict.fromkeys(self.agents, 0.0), False,
                     self.task, roles, {agent: list(self.history) for agent in actors}, {})

    def step(self, actions):
        expected = self.frame()
        if set(actions) != set(expected.legal_actions):
            raise ValueError("Avalon transition has different acting participants")
        if any(action not in expected.legal_actions[agent] for agent, action in actions.items()):
            raise ValueError("Avalon transition contains an illegal action")
        phase = int(self.engine.phase)
        if phase == 0:
            agent, action = next(iter(actions.items()))
            team = self.team_catalog[action]
            self.engine.choose_quest_team(team, agent)
            self.history.append({"leader": agent, "team": sorted(team)})
        elif phase == 1:
            votes = [int(actions[agent] == self.vote_approve) for agent in self.agents]
            self.engine.gather_team_votes(votes)
            self.history.append({"team_votes": votes})
        elif phase == 2:
            votes = [int(actions[agent] == self.quest_success) for agent in sorted(actions)]
            self.engine.gather_quest_votes(votes)
            self.history.append({"quest_results": [bool(value) for value in self.engine.quest_results]})
        else:
            agent, action = next(iter(actions.items()))
            self.engine.choose_assassination_target(agent, action - self.assassination_start)
        return self.frame()

    def close(self):
        pass


class DecryptoEnvironment:
    def __init__(self, clue_catalog, source="external/decrypto", expected_revision=None,
                 max_turns=8, condition="competitive_interception"):
        source = Path(source)
        self.version = source_revision(source, expected_revision)
        specification = importlib.util.spec_from_file_location("native_decrypto", source / "src/env/decrypto.py")
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        self.engine = module.Decrypto(max_turns=max_turns)
        self.clues = json.loads(Path(clue_catalog).read_text())
        self.clues = {int(identifier): value for identifier, value in self.clues.items()}
        if not self.clues or any(len(value["native"]) != 3 for value in self.clues.values()):
            raise ValueError("Decrypto action catalog contains three-clue native actions")
        self.guess_start = max(self.clues) + 1
        self.guesses = list(itertools.permutations(range(1, 5), 3))
        self.agents = (0, 1, 2)
        self.name = "Decrypto"
        self.condition = condition
        self.task = ("Use the released Decrypto rules. The encoder and decoder seek successful communication; "
                     "the interceptor seeks successful interception. Condition " + condition + ".")

    def reset(self, seed):
        self.code, self.keywords, self.info = self.engine.reset(seed)
        self.phase = "encoder"
        self.hints = None
        return self.frame()

    def frame(self, rewards=None):
        rewards = rewards or dict.fromkeys(self.agents, 0.0)
        if self.engine.done:
            return Frame({}, {}, rewards, True, self.task, {}, {},
                         {"success": 100 * float(rewards[0] > 0)})
        public = {"turn": int(self.info["turn"]), "miscommunications": int(self.info["miscommunications"]),
                  "interceptions": int(self.info["interceptions"]),
                  "code_history": [np.asarray(value).tolist() for value in self.info["code history"]],
                  "hint_history": self.info["hint history"]}
        if self.phase == "encoder":
            observation = dict(public, keywords=list(self.keywords), code=self.code)
            legal = {identifier: value["meaning"] for identifier, value in self.clues.items()}
            return Frame({0: observation}, {0: legal}, rewards, False, self.task, {0: "encoder"}, {}, {})
        legal = {self.guess_start + index: "Guess " + "".join(map(str, guess)) for index, guess in enumerate(self.guesses)}
        return Frame({1: dict(public, keywords=list(self.keywords), hints=self.hints),
                      2: dict(public, hints=self.hints)}, {1: legal, 2: legal}, rewards, False,
                     self.task, {1: "decoder", 2: "interceptor"}, {}, {})

    def step(self, actions):
        expected = self.frame()
        if set(actions) != set(expected.legal_actions):
            raise ValueError("Decrypto transition has different acting participants")
        if any(action not in expected.legal_actions[agent] for agent, action in actions.items()):
            raise ValueError("Decrypto transition contains an illegal action")
        if self.phase == "encoder":
            self.hints = list(self.clues[actions[0]]["native"])
            self.phase = "guesses"
            return self.frame()
        decoder = np.asarray(self.guesses[actions[1] - self.guess_start])
        interceptor = np.asarray(self.guesses[actions[2] - self.guess_start])
        self.code, rewards, terminated, self.info = self.engine.step(self.hints, decoder, interceptor)
        self.phase = "encoder"
        return self.frame({0: float(rewards["alice"]), 1: float(rewards["bob"]), 2: float(rewards["eve"])})

    def close(self):
        pass
