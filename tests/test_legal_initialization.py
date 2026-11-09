import random
import unittest
from dataclasses import replace
from unittest.mock import patch
from common.model import Excursion
from stage2.decoder import Decoder
from stage2.initialization import construct, draft_check
from stage2.initial_capacity import CapacityGuard
from stage2.constructive_repair import compatible, repair
from stage2.routing import Infeasible
from tests.fixtures import scenario


class LegalInitializationTests(unittest.TestCase):
    def test_legal_first_station_split_and_independent_continuation_are_searchable(self):
        n, p, table = scenario(ordinary=1, drt=1)
        n.passengers = [replace(q, latest=.25) if q.kind else q for q in n.passengers]
        decoder = Decoder(n, table, p)
        plan = construct(decoder, 0, random.Random(p.seed), progress=lambda _: None)
        self.assertEqual({q for e in plan for q, pickup in e.events if pickup}, {100})
        self.assertTrue(any(e.split == 0 for e in plan))
        self.assertTrue(all(e.target in decoder.nominal_trips or e.target.startswith('backup:') for e in plan))
        self.assertTrue(compatible(decoder, plan))
        result = decoder.decode(plan, {100}, audit=True)
        self.assertFalse(result.errors, result.errors)
        self.assertEqual(set(result.services), {1, 100})

    def test_optimistic_segment_supply_cannot_replace_complete_boarding_check(self):
        n, p, _ = scenario(ordinary=7)
        n.lines = {'A': n.lines['A'], 'Ar': n.lines['Ar']}
        n.passengers = [replace(q, origin=1) if not q.kind else q for q in n.passengers]
        n.nodes[12] = replace(n.nodes[12], x=0, y=0)
        n.nodes[13] = replace(n.nodes[13], x=500, y=0)
        p = replace(p, max_formation=2)
        decoder = Decoder(n, {'A': [0], 'Ar': [0]}, p)
        decision = Excursion('A:0/m1', 0, 'backup:A', 2, ((100, True), (100, False)), 'A:0', 0)
        context = decoder.context((decision,))
        CapacityGuard(n, p).check(context.trips, context.duties, (decision,))
        with self.assertRaisesRegex(Infeasible, 'complete ordinary boarding'):
            draft_check(decoder, (decision,))

    def test_continuation_plan_stays_in_fast_repair_and_uses_final_joint_audit(self):
        n, p, table = scenario(ordinary=2, drt=2)
        decoder = Decoder(n, table, p)
        plan = construct(decoder, 0, random.Random(p.seed), progress=lambda _: None)
        self.assertTrue(all(e.target in decoder.nominal_trips or e.target.startswith('backup:') for e in plan))
        with patch.object(decoder, 'decode', wraps=decoder.decode) as allocate:
            result, failure = repair(decoder, plan, [], 0, random.Random(p.seed))
        self.assertIsNone(failure)
        self.assertFalse(result.errors, result.errors)
        self.assertEqual(allocate.call_count, 1)
        self.assertTrue(allocate.call_args.kwargs['audit'])
        self.assertIsNone(decoder.optimizer.initial_hint)
        self.assertEqual(set(result.services), {1, 2, 100, 101})
