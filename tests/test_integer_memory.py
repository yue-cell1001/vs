import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from scipy.sparse import csr_array
from common.integer import solve_integer

class IntegerMemoryTests(unittest.TestCase):

    def setUp(self):
        self.args = ([1.0], [0.0], [2.0], csr_array([[1.0]]), [1.0], [2.0])

    def test_native_memory_is_released_and_verified_incumbent_survives(self):
        h = MagicMock()
        h.run.side_effect = MemoryError('bad allocation')
        with patch('common.integer.Highs', return_value=h):
            result = solve_integer(*self.args, warm=[1.0])
        h.clear.assert_called_once()
        np.testing.assert_array_equal(result.x, [1.0])
        self.assertFalse(result.optimal)
        self.assertIsNone(result.gap)
        self.assertIn('memory_limit', result.message)

    def test_invalid_or_missing_incumbent_never_accepted(self):
        for warm in (None, [0.0], [3.0], [1.5], [np.nan], [1.0, 1.0]):
            with patch('common.integer._solve_integer', side_effect=MemoryError()):
                with self.assertRaisesRegex(MemoryError, 'no verified incumbent'):
                    solve_integer(*self.args, warm=warm)

    def test_real_solver_still_solves_optimally(self):
        result = solve_integer(*self.args)
        self.assertTrue(result.optimal)
        np.testing.assert_array_equal(result.x, [1.0])
