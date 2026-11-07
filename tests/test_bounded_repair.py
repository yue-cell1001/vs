import unittest
from dataclasses import replace
from time import monotonic
from unittest.mock import patch
from common.integer import solve_integer
from stage2.alns import solve
from stage2.budget import SearchTimeout
from stage2.decoder import Decoder
from stage2.routing import Infeasible
from tests.fixtures import scenario


class BoundedRepairTests(unittest.TestCase):
    def test_old_preference_labels_share_non_audited_joint_solve(self):
        n, p, table = scenario(ordinary=1, drt=0)
        decoder = Decoder(n, table, p)
        with patch.object(decoder.optimizer, 'solve', wraps=decoder.optimizer.solve) as allocate:
            first = decoder.decode((), set())
            decoder.preference = 1
            second = decoder.decode((), set())
            second.services.clear()
            third = decoder.decode((), set())
        self.assertEqual(allocate.call_count, 1)
        self.assertFalse(first.errors)
        self.assertEqual(third.objectives, first.objectives)
        self.assertEqual(len(third.services), 1)
        self.assertEqual(third.metadata['allocation_preference'], 1)

    def test_joint_solve_uses_remaining_repair_budget(self):
        n, p, table = scenario(ordinary=1, drt=0)
        decoder = Decoder(n, table, replace(p, subproblem_seconds=30))
        decoder.deadline = monotonic() + 5
        with patch('stage2.joint.solve_integer', wraps=solve_integer) as allocate:
            result = decoder.decode((), set(), audit=True)
        self.assertFalse(result.errors, result.errors)
        self.assertGreater(allocate.call_args.args[6], 0)
        self.assertLessEqual(allocate.call_args.args[6], 5)

    def test_expired_repair_cannot_start_another_joint_solve(self):
        n, p, table = scenario()
        decoder = Decoder(n, table, p)
        decoder.deadline = monotonic() - 1
        with patch('stage2.joint.solve_integer') as allocate:
            with self.assertRaises(SearchTimeout):
                decoder.decode((), set())
        allocate.assert_not_called()

    def test_timed_out_iterations_keep_initial_solution_and_are_not_feasible_repairs(self):
        n, p, table = scenario()
        with patch('stage2.alns.repair', side_effect=SearchTimeout('Search time budget exhausted')) as repair:
            items, stats = solve(n, table, replace(p, iterations=2), progress=lambda _: None)
        self.assertTrue(items)
        self.assertEqual(repair.call_count, 2)
        self.assertEqual(stats['complete_feasible_iterations'], 0)
        self.assertTrue(all(row['repair_timed_out'] for row in stats['trace']))
        self.assertTrue(all(not row['repair_complete'] for row in stats['trace']))
        self.assertTrue(all(not item.errors for item in items))

    def test_failed_uploaded_initialization_never_calls_original_constructor(self):
        from stage2.alns import NoFeasibleSolution
        n, p, table = scenario()
        with patch('stage2.initialization.construct', side_effect=Infeasible('Uploaded construction rejected')) as uploaded, patch('stage2.initialization._construct_preserved_repair', side_effect=AssertionError('Original initialization must not run'), create=True) as original:
            with self.assertRaises(NoFeasibleSolution) as caught:
                solve(n, table, replace(p, iterations=0), progress=lambda _: None)
        self.assertEqual(uploaded.call_count, 1)
        original.assert_not_called()
        attempts = caught.exception.diagnostics['initialization_attempts']
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]['strategy'], 'uploaded')
