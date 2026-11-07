import random
import unittest
from dataclasses import replace
from unittest.mock import patch
from stage2.alns import solve, NoFeasibleSolution
from stage2.budget import SearchTimeout
from stage2.decoder import Decoder
from stage2.initialization import construct
from stage2.constructive_repair import compatible, repair
from stage2.learning import AdaptiveAutomaton, ACTIONS
from tests.fixtures import scenario


class ScheduledSearchTests(unittest.TestCase):
    def test_generated_initial_plan_is_compatible_with_single_allocation_repairs(self):
        n, p, table = scenario(ordinary=2, drt=2)
        decoder = Decoder(n, table, p)
        plan = construct(decoder, 0, random.Random(p.seed), progress=lambda _: None)
        self.assertTrue(compatible(decoder, plan))
        self.assertTrue(all(e.target in decoder.nominal_trips for e in plan))
        with patch.object(decoder.optimizer, 'solve', wraps=decoder.optimizer.solve) as allocate:
            result, failure = repair(decoder, plan, [], 0, random.Random(p.seed))
        self.assertIsNone(failure)
        self.assertFalse(result.errors)
        self.assertEqual(allocate.call_count, 1)
        self.assertEqual(result.metadata['joint_solver']['solve_calls_this_allocation'], 1)
        self.assertEqual(result.metadata['joint_solver']['optimization_method'], 'augmented_weighted_chebyshev')
        self.assertEqual(set(result.services), {1, 2, 100, 101})

    def test_adaptive_learning_changes_state_and_restores_exactly(self):
        learner = AdaptiveAutomaton()
        self.assertEqual(len(ACTIONS), 15)
        for _ in range(3):
            action = learner.choose(random.Random(1))
            learner.feedback(action, 0, feasible=False)
        self.assertEqual(learner.state(), 1)
        for _ in range(40):
            action = learner.choose(random.Random(2))
            learner.feedback(action, 0, feasible=True)
        self.assertEqual(learner.state(), 2)
        action = learner.choose(random.Random(3))
        learner.feedback(action, .5, feasible=True, hv_gain=.1)
        self.assertEqual(learner.state(), 0)
        restored = AdaptiveAutomaton()
        restored.restore(learner.snapshot())
        self.assertEqual(restored.snapshot(), learner.snapshot())
        self.assertEqual(restored.choose(random.Random(4)), learner.choose(random.Random(4)))

    def test_initialization_timeout_is_saved_as_failure_and_never_enters_iteration(self):
        n, p, table = scenario()
        with patch('stage2.initialization.construct', side_effect=SearchTimeout('Search time budget exhausted')), patch('stage2.alns.repair') as repair:
            with self.assertRaises(NoFeasibleSolution) as caught:
                solve(n, table, replace(p, iterations=1), progress=lambda _: None)
        repair.assert_not_called()
        self.assertTrue(caught.exception.diagnostics['initial_failures'][0]['timed_out'])

    def test_scheduling_destroy_can_pair_with_all_three_repairs(self):
        for rule in range(3):
            n, p, table = scenario()
            with patch('stage2.alns.Automaton.choose', return_value=(4, rule)), patch('stage2.alns.repair', return_value=(None, 'test')) as repair:
                items, stats = solve(n, table, replace(p, iterations=1), progress=lambda _: None)
            self.assertTrue(items)
            self.assertEqual(repair.call_count, 1)
            self.assertEqual(stats['trace'][0]['action'], (4, rule))
