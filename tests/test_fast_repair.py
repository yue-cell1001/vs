import random
import unittest
from unittest.mock import patch
from stage2.decoder import Decoder
from stage2.constructive_repair import repair
from tests.fixtures import scenario


class FastRepairTests(unittest.TestCase):

    def test_repair_solves_joint_model_only_after_all_requests_are_reinserted(self):
        for rule in (0, 1, 2):
            with self.subTest(rule=rule):
                n, p, table = scenario(ordinary=7, drt=3)
                decoder = Decoder(n, table, p)
                with patch.object(decoder, 'decode', wraps=decoder.decode) as allocate:
                    result, failure = repair(decoder, (), [100, 101, 102], rule, random.Random(p.seed))
                self.assertIsNone(failure)
                self.assertIsNotNone(result)
                self.assertFalse(result.errors)
                self.assertEqual(len(result.services), 10)
                self.assertEqual(allocate.call_count, 1)
                self.assertTrue(allocate.call_args.kwargs['audit'])
                self.assertEqual(allocate.call_args.args[1], {100, 101, 102})
                self.assertIn(result.metadata['constructive_repair_rule'], (0, 1, 2))

    def test_verified_decode_cache_reuses_solves_without_sharing_mutable_results(self):
        n, p, table = scenario(ordinary=1, drt=0)
        decoder = Decoder(n, table, p)
        with patch.object(decoder.optimizer, 'solve', wraps=decoder.optimizer.solve) as solve:
            first = decoder.decode((), set(), audit=True)
            self.assertFalse(first.errors)
            first.services.clear()
            second = decoder.decode((), set(), audit=True)
            self.assertEqual(solve.call_count, 1)
            self.assertEqual(len(second.services), 1)
            self.assertEqual(decoder.decode_cache_hits, 1)
            decoder.preference = 1
            third = decoder.decode((), set(), audit=True)
            self.assertFalse(third.errors)
            self.assertEqual(solve.call_count, 1)

    def test_repair_preserves_existing_drt_transfer_and_adds_pending_request(self):
        from dataclasses import replace
        from common.model import Excursion
        n, p, table = scenario(ordinary=5, drt=2)
        n.nodes[12] = replace(n.nodes[12], x=1000, y=1100)
        n.nodes[13] = replace(n.nodes[13], x=1750, y=750)
        n.nodes[8] = replace(n.nodes[6], id=8, x=500, y=1500)
        n.lines['B'] = (4, 1, 5, 6, 8)
        n.lines['Br'] = (8, 6, 5, 1, 4)
        n.edges[6, 8] = n.edges[8, 6] = (0.5, 1.0)
        partial = (
            Excursion('A:0/m1', 1, 'B:1', 1, ((100, True),), 'A:0', 1),
            Excursion('B:1/m2', 2, 'B:2', 3, ((100, False),), 'B:1', 5),
        )
        decoder = Decoder(n, table, p)
        result, failure = repair(decoder, partial, [101], 0, random.Random(p.seed))
        self.assertIsNone(failure)
        self.assertFalse(result.errors)
        self.assertEqual(len(result.services), 7)
        self.assertEqual(len(result.services[100].rides), 2)
