import random
import unittest
from dataclasses import replace
from unittest.mock import patch
from stage2.alns import solve
from stage2.decoder import Decoder
from stage2.initialization import construct
from tests.fixtures import scenario


class SimpleIterationTests(unittest.TestCase):

    def test_failed_repair_is_attempted_only_once_per_iteration(self):
        n, p, table = scenario()
        with patch('stage2.alns.repair', return_value=(None, {'reason': 'No insertion'})) as repair:
            solutions, stats = solve(n, table, replace(p, iterations=3), progress=lambda _: None)
        self.assertTrue(solutions)
        self.assertEqual(repair.call_count, 3)
        self.assertEqual(stats['iterations'], 3)
        self.assertEqual(len(stats['trace']), 3)
        self.assertTrue(all(not row['repair_complete'] for row in stats['trace']))
        self.assertEqual(sum(stats['learner']['selections']), 3)

    def test_synchronous_repair_tries_one_legal_shift_and_stops_on_failure(self):
        n, p, table = scenario()
        table = {line: [0, 7, 15] for line in table}
        with patch('stage2.alns.Automaton.choose', return_value=(4, 2)), patch('stage2.alns.repair', return_value=(None, {'reason': 'No insertion'})) as repair:
            solutions, stats = solve(n, table, replace(p, iterations=1), progress=lambda _: None)
        self.assertEqual(repair.call_count, 1)
        attempted = repair.call_args.args[0].timetable
        self.assertNotEqual(attempted, table)
        from stage2.timing import validate_branch
        self.assertTrue(validate_branch(attempted, table, p))
        self.assertEqual(stats['iterations'], 1)
        self.assertFalse(stats['trace'][0]['repair_complete'])
        self.assertTrue(solutions)

    def test_original_repair_preserves_deadline_order_for_both_scoring_rules(self):
        from stage2.initialization import sequence
        n, p, table = scenario(ordinary=1, drt=3)
        for rule in (0, 1):
            decoder = Decoder(n, table, p)
            seen = []
            def observed(*args):
                for pid, pickup in args[2]:
                    if pickup and pid not in seen:
                        seen.append(pid)
                return sequence(*args)
            with patch('stage2.initialization.sequence', side_effect=observed), patch.object(decoder, 'decode', side_effect=AssertionError('Route construction must not run the joint model')):
                plan = construct(decoder, rule, random.Random(p.seed), progress=lambda _: None, pending=[102, 100, 101])
            self.assertEqual(seen, [100, 101, 102])
            result = decoder.decode(plan, {100, 101, 102}, audit=True)
            self.assertFalse(result.errors, result.errors)
