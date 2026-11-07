from copy import deepcopy
from math import isfinite, isclose

ACTIONS = tuple((d, r) for d in range(5) for r in range(3))


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


class AdaptiveAutomaton(Automaton):
    def __init__(self, actions=ACTIONS, rate=0.1, exploration=0.05):
        super().__init__(actions, rate, exploration)
        self.state_probabilities = [self.probabilities[:] for _ in range(3)]
        self.failures = []
        self.no_gain = 0
        self.last_state = 0

    def state(self):
        if self.failures and sum(self.failures) / len(self.failures) >= .6:
            return 1
        return 2 if self.no_gain >= 40 else 0

    def choose(self, rng, available=None):
        self.last_state = self.state()
        self.probabilities = self.state_probabilities[self.last_state]
        original = self.exploration
        self.exploration = (original, max(original, .25), max(original, .2))[self.last_state]
        try:
            return super().choose(rng, available)
        finally:
            self.exploration = original

    def feedback(self, action, reward, feasible=True, hv_gain=0):
        super().feedback(action, reward)
        self.state_probabilities[self.last_state] = self.probabilities[:]
        self.failures = (self.failures + [not feasible])[-30:]
        self.no_gain = 0 if hv_gain > 1e-15 else self.no_gain + 1

    def snapshot(self):
        result = super().snapshot()
        result.update(schema='adaptive-lri-v1', state_probabilities=deepcopy(self.state_probabilities),
                      failures=self.failures[:], no_gain=self.no_gain, last_state=self.last_state)
        return result

    def restore(self, state):
        if state.get('schema') != 'adaptive-lri-v1':
            raise ValueError('Checkpoint uses a different learning algorithm')
        matrices = state.get('state_probabilities', [])
        if len(matrices) != 3 or state.get('last_state') not in (0, 1, 2):
            raise ValueError('Invalid adaptive learning checkpoint')
        for row in matrices:
            super().restore({**state, 'schema': 'unified-lri-v1', 'probabilities': row})
        super().restore({**state, 'schema': 'unified-lri-v1'})
        if (len(state['failures']) > 30 or any(type(v) is not bool for v in state['failures'])
                or type(state['no_gain']) is not int or state['no_gain'] < 0
                or self.probabilities != matrices[state['last_state']]):
            raise ValueError('Invalid adaptive learning history')
        self.state_probabilities = deepcopy(matrices)
        self.failures = state['failures'][:]
        self.no_gain = state['no_gain']
        self.last_state = state['last_state']
