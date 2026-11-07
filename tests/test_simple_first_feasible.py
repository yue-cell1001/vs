import random
import unittest
from unittest.mock import patch
from stage2.decoder import Decoder
from stage2.initialization import construct, draft_check
from stage2.alns import repair
from tests.fixtures import scenario


class SimpleFirstFeasibleTests(unittest.TestCase):
    def test_urgent_repair_stops_at_first_passing_candidate_per_request(self):
        n, p, table = scenario(ordinary=2, drt=3)
        decoder = Decoder(n, table, p)
        with patch('stage2.initialization.draft_check', wraps=draft_check) as check:
            plan = construct(decoder, 1, random.Random(p.seed), progress=lambda _: None, pending=[102, 100, 101])
        self.assertEqual(check.call_count, 3)
        self.assertEqual(decoder.initialization_stats['request_order'], [100, 101, 102])
        self.assertEqual(decoder.initialization_stats['strategy'], 'urgent_first_feasible')
        result = decoder.decode(plan, {100, 101, 102}, audit=True)
        self.assertFalse(result.errors, result.errors)
        self.assertEqual(len(result.services), 5)

    def test_greedy_repair_keeps_supplied_order_and_still_serves_every_request(self):
        n, p, table = scenario(ordinary=2, drt=3)
        decoder = Decoder(n, table, p)
        plan = construct(decoder, 0, random.Random(p.seed), progress=lambda _: None, pending=[102, 100, 101])
        self.assertEqual(decoder.initialization_stats['request_order'], [102, 100, 101])
        result = decoder.decode(plan, {100, 101, 102}, audit=True)
        self.assertFalse(result.errors, result.errors)
        self.assertEqual(len(result.services), 5)

    def test_default_repair_never_silently_switches_to_exhaustive_insertion(self):
        n, p, table = scenario()
        decoder = Decoder(n, table, p)
        with patch('stage2.constructive_repair.compatible', return_value=False), patch('stage2.alns.insert') as exhaustive:
            result, failure = repair(decoder, (), [100], 0, random.Random(p.seed))
        self.assertIsNone(result)
        self.assertEqual(failure['reason'], 'unsupported_constructive_plan')
        exhaustive.assert_not_called()
