import unittest
from unittest.mock import patch
import numpy as np
from scipy.sparse import csr_array
from common.integer import solve_integer, _verified, Result
from stage2.scalarization import build_problem, reference_scales, DIRECTIONS
from stage2.decoder import Decoder
from tests.fixtures import scenario


def allocation(c1, c2, scales=(10, 10), weights=(.5, .5), constant=0):
    n = len(c1)
    problem = build_problem(c1, c2, constant, [0] * n, [1] * n,
                            csr_array(np.ones((1, n))), [1], [1], None, scales, weights)
    cost, lo, hi, matrix, lb, ub, warm, integer, _ = problem
    result = solve_integer(cost, lo, hi, matrix, lb, ub, warm=warm, integrality=integer)
    return result, int(np.argmax(result.x[:-1]))


class SymmetricScalarizationTests(unittest.TestCase):
    def test_equal_direction_finds_unsupported_discrete_pareto_compromise(self):
        result, choice = allocation([1, 6, 10], [10, 6, 1])
        self.assertTrue(result.optimal)
        self.assertEqual(choice, 1)
        self.assertGreater(6 + 6, 1 + 10)

    def test_direction_set_is_symmetric_and_contains_equal_weights(self):
        self.assertIn((.5, .5), DIRECTIONS)
        self.assertTrue(all((b, a) in DIRECTIONS for a, b in DIRECTIONS))
        for w, expected in [((.8, .2), 0), ((.5, .5), 1), ((.2, .8), 2)]:
            self.assertEqual(allocation([1, 6, 10], [10, 6, 1], weights=w)[1], expected)

    def test_exchanging_objectives_and_weights_preserves_choice(self):
        for weights in DIRECTIONS:
            original = allocation([1, 6, 10], [10, 6, 1], weights=weights)[1]
            exchanged = allocation([10, 6, 1], [1, 6, 10], weights=weights[::-1])[1]
            self.assertEqual(original, exchanged)

    def test_unit_changes_with_matching_scales_do_not_change_choice(self):
        original = allocation([1, 6, 10], [10, 6, 1])[1]
        self.assertEqual(allocation([1000, 6000, 10000], [10, 6, 1], scales=(10000, 10))[1], original)
        self.assertEqual(allocation([1, 6, 10], [600, 360, 60], scales=(10, 600))[1], original)

    def test_constant_operating_cost_is_included_in_max_loss(self):
        result, _ = allocation([2], [3], scales=(1, 1), constant=5)
        self.assertTrue(result.optimal)
        self.assertAlmostEqual(result.x[-1], 3.5e6)

    def test_continuous_auxiliary_variable_is_not_rounded_to_integer(self):
        matrix = csr_array([[1., 0.], [0., 1.]])
        result = solve_integer([1, 1], [0, 0], [2, 2], matrix, [1, .25], [2, 2], integrality=[1, 0])
        self.assertTrue(result.optimal)
        np.testing.assert_allclose(result.x, [1, .25])
        self.assertIsNone(_verified([1.2, .25], [0, 0], [2, 2], matrix, [1, .25], [2, 2], [1, 0]))
        self.assertIsNone(_verified([1, .1], [0, 0], [2, 2], matrix, [1, .25], [2, 2], [1, 0]))

    def test_reference_scales_are_frozen_from_both_reference_objectives(self):
        self.assertEqual(reference_scales([{'objectives': [50, 200]}, {'objectives': [100, 80]}], (900, 900)), (100, 200))
        self.assertEqual(reference_scales([{'timetable': {}}], (900, 1000)), (900, 1000))
        with self.assertRaises(ValueError):
            reference_scales([{'objectives': [float('nan'), 1]}], (1, 1))

    def test_cache_is_keyed_by_direction_and_scale(self):
        n, p, table = scenario(ordinary=1, drt=0)
        decoder = Decoder(n, table, p)
        with patch.object(decoder.optimizer, 'solve', wraps=decoder.optimizer.solve) as solve:
            first = decoder.decode((), set(), audit=True)
            self.assertFalse(first.errors)
            decoder.decode((), set(), audit=True)
            self.assertEqual(solve.call_count, 1)
            decoder.tradeoff_weights = (.2, .8)
            second = decoder.decode((), set(), audit=True)
            self.assertFalse(second.errors)
            self.assertEqual(second.metadata['joint_solver']['tradeoff_weights'], [.2, .8])
            self.assertEqual(solve.call_count, 2)
            decoder.objective_scales = (100, 200)
            decoder.decode((), set(), audit=True)
            self.assertEqual(solve.call_count, 3)

    def test_time_limited_feasible_result_is_not_reported_scalar_optimal(self):
        n, p, table = scenario(ordinary=1, drt=0)
        def limited(*args, **kwargs):
            result = solve_integer(*args, **kwargs)
            return Result(result.x, False, .2, result.lower_bound, 'Time limit reached')
        with patch('stage2.joint.solve_integer', side_effect=limited) as solve:
            solution = Decoder(n, table, p).decode((), set(), audit=True)
        self.assertFalse(solution.errors)
        self.assertEqual(solve.call_count, 1)
        self.assertEqual(solution.metadata['joint_solver']['status'], 'feasible_not_proven_optimal')
        self.assertEqual(solution.metadata['joint_solver']['scalar_gap'], .2)
