"""One linear reward-inaction automaton, with fixed uniform exploration."""
from copy import deepcopy
from math import isfinite, isclose

ACTIONS = tuple((d, r) for d in range(5) for r in range(3) if d != 4 or r == 2)


class Automaton:
    def __init__(self, actions=ACTIONS, rate=0.1, exploration=0.05):
        self.actions = list(actions)
        if not self.actions or len(set(self.actions)) != len(self.actions):
            raise ValueError('Actions must be nonempty and unique')
        if not 0 < rate < 1 or not 0 <= exploration <= 1:
            raise ValueError('Invalid learning parameters')
        self.rate, self.exploration = rate, exploration
        self.probabilities = [1 / len(self.actions)] * len(self.actions)
        self.selections = [0] * len(self.actions)
        self.rewards = [0.0] * len(self.actions)

    def sampling_probabilities(self, available=None):
        available = self.actions if available is None else list(available)
        if not available or len(set(available)) != len(available):
            raise ValueError('Available actions must be nonempty and unique')
        ids = [self.actions.index(action) for action in available]
        weights = [self.probabilities[i] for i in ids]
        total = sum(weights)
        if not total:
            return [1 / len(ids)] * len(ids)
        eps = self.exploration
        return [(1 - eps) * w / total + eps / len(ids) for w in weights]

    def choose(self, rng, available=None):
        available = self.actions if available is None else list(available)
        weights = self.sampling_probabilities(available)
        action = rng.choices(available, weights=weights, k=1)[0]
        self.selections[self.actions.index(action)] += 1
        return action

    def feedback(self, action, reward):
        if not isfinite(reward) or not 0 <= reward <= 1:
            raise ValueError('Invalid reward')
        j = self.actions.index(action)
        self.rewards[j] += reward
        if reward == 0:
            return
        factor = self.rate * reward
        self.probabilities = [(1 - factor) * p + (factor if k == j else 0)
                              for k, p in enumerate(self.probabilities)]

    def snapshot(self):
        return deepcopy({'schema': 'unified-lri-v1', 'actions': self.actions,
                         'probabilities': self.probabilities, 'selections': self.selections,
                         'rewards': self.rewards})

    def restore(self, state):
        actions = [tuple(a) for a in state.get('actions', ())]
        if state.get('schema') != 'unified-lri-v1' or actions != self.actions:
            raise ValueError('Checkpoint uses a different learning algorithm/action set')
        n = len(self.actions)
        probabilities = state['probabilities']
        selections, rewards = state['selections'], state['rewards']
        if (len(probabilities) != n or len(selections) != n or len(rewards) != n
                or any(not isfinite(v) or v < 0 for v in probabilities)
                or not isclose(sum(probabilities), 1, abs_tol=1e-10)
                or any(type(v) is not int or v < 0 for v in selections)
                or any(not isfinite(v) or v < 0 for v in rewards)):
            raise ValueError('Invalid learning checkpoint')
        self.probabilities = list(probabilities)
        self.selections = list(selections)
        self.rewards = list(rewards)
