import random
import unittest
from copy import deepcopy
from common.multiobjective import layers2d, hypervolume
from common.pareto import fronts, Archive
from common.model import Solution
from stage2.learning import Automaton
from stage2.events import EventNetwork

class MultiobjectiveTests(unittest.TestCase):

    def test_fast_layers_equal_pairwise_sort(self):
        rng = random.Random(3)
        values = [(rng.randrange(15), rng.randrange(15)) for _ in range(100)]
        self.assertEqual([set(f) for f in layers2d(values)], [set(f) for f in fronts(values)])

    def test_roundoff_does_not_create_pareto_tradeoff(self):
        values = [(10.0, 20.0), (10.0 + 1e-12, 19.0)]
        self.assertEqual(layers2d(values)[0], [1])
        self.assertEqual(fronts(values)[0], [1])

    def test_hv_exact_and_duplicate_invariance(self):
        points = [(1, 4), (2, 2), (4, 1)]
        self.assertAlmostEqual(hypervolume(points, (5, 5), 0.1), 0.61)
        self.assertAlmostEqual(hypervolume(points + points + [(4, 4)], (5, 5), 0.1), 0.61)
        with self.assertRaises(ValueError):
            hypervolume([(6, 1)], (5, 5), 0.1)

    def test_unified_learning_and_exact_reward_inaction(self):
        a = Automaton()
        self.assertEqual(len(a.actions), 13)
        self.assertEqual(a.probabilities, [1 / 13] * 13)
        before = a.probabilities[:]
        a.feedback((0, 0), 0)
        self.assertEqual(a.probabilities, before)
        a.feedback((1, 0), 0.3)
        for i, value in enumerate(a.probabilities):
            self.assertAlmostEqual(value, .97 / 13 + (.03 if a.actions[i] == (1, 0) else 0))
        self.assertAlmostEqual(sum(a.probabilities), 1)

    def test_available_action_and_snapshot(self):
        a = Automaton()
        action = a.choose(random.Random(1), [(1, 0)])
        self.assertEqual(action, (1, 0))
        a.feedback(action, 1)
        b = Automaton()
        b.restore(a.snapshot())
        self.assertEqual(a.snapshot(), b.snapshot())
        snapshot = a.snapshot()
        snapshot['probabilities'][0] = 99
        self.assertNotEqual(a.probabilities[0], 99)

    def test_fixed_exploration_is_applied_only_when_sampling(self):
        a = Automaton([(0, 0), (1, 0)], exploration=.05)
        a.probabilities = [.8, .2]
        weights = a.sampling_probabilities()
        self.assertAlmostEqual(weights[0], .95 * .8 + .05 / 2)
        self.assertAlmostEqual(weights[1], .95 * .2 + .05 / 2)
        self.assertEqual(a.sampling_probabilities([(1, 0)]), [1])
        for _ in range(1000):
            a.feedback((0, 0), 1)
        self.assertGreaterEqual(a.sampling_probabilities()[1], .05 / 2)
        self.assertLess(a.probabilities[1], .001)
        with self.assertRaises(ValueError):
            a.restore({'probabilities': [[1 / 13] * 13] * 3})

    def test_event_cycle_rejected_and_visits_distinct(self):
        g = EventNetwork()
        g.bound(('v', 0), 0, 0)
        g.equal(('v', 0), ('v', 1), 3)
        self.assertEqual(g.solve()['v', 1], 3)
        g.difference(('v', 1), ('v', 0), lower=1)
        with self.assertRaises(ValueError):
            g.solve()
