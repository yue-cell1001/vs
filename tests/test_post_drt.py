import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from common.model import Excursion
from stage2.decoder import Decoder
from stage2.constraints import validate
from stage2.obj import evaluate
from stage2.plot import draw, segments
from tests.fixtures import scenario

class PostDRTTests(unittest.TestCase):

    def decision(self, target, join=2, events=((100, True), (100, False))):
        return Excursion('A:0/m1', 1, target, join, events, 'A:0', 1)

    def test_solo_retains_ordinary_rider_and_full_solo_cost(self):
        n, p, table = scenario(ordinary=7)
        p = replace(p, max_formation=2)
        result = Decoder(n, table, p).decode((self.decision('backup:A'),), {100}, audit=True)
        self.assertFalse(result.errors, result.errors)
        duty = result.duties['A:0/m1']
        solo = [leg for leg in duty.legs if leg.kind == 'solo']
        self.assertTrue(solo)
        rows = [row for row in segments(result, n, p) if row['trip'].startswith(('solo/', 'backup/'))]
        self.assertEqual(len(rows), len(solo))
        self.assertTrue(all(row['formation'] == 1 for row in rows))
        self.assertTrue(all(row['trip'] != 'A:0' for row in rows))
        self.assertTrue(any(not next(q for q in n.passengers if q.id == pid).kind and service.rides[0].duty == duty.id for pid, service in result.services.items()))
        self.assertAlmostEqual(result.components['solo_travel'], sum(leg.distance for leg in solo) * p.operating_cost)
        self.assertEqual(evaluate(result, n, p), result.objectives)
        self.assertFalse(validate(result, n, None, p))
        with tempfile.TemporaryDirectory() as tmp:
            draw(result, n, p, tmp, dpi=40)
            self.assertTrue((Path(tmp) / 'A.png').exists())
        duty.legs[-1] = replace(duty.legs[-1], distance=duty.legs[-1].distance + 1)
        self.assertTrue(validate(result, n, None, p))

    def test_fixed_donor_can_return_to_warehouse(self):
        n, p, table = scenario()
        result = Decoder(n, table, p).decode((self.decision('depot:1', 0),), {100}, audit=True)
        self.assertFalse(result.errors, result.errors)
        self.assertEqual(result.duties['A:0/m1'].visits[-1].kind, 'depot')
        self.assertTrue(any(entry['kind'] == 'final' or entry['kind'] == 'between_duties' for entry in result.metadata['storage']))

    def test_waiting_is_not_coupled_running(self):
        n, p, table = scenario()
        n.passengers = [replace(q, latest=3) if q.kind else q for q in n.passengers]
        decoder = Decoder(n, table, p)
        decision = self.decision('A:2')
        result = decoder.decode((decision,), {100}, audit=True)
        self.assertFalse(result.errors, result.errors)
        duty = result.duties[decision.source]
        join = next(v for v in duty.visits if v.trip == 'A:2' and v.kind == 'join')
        self.assertGreater(join.departure - join.arrival, 1)
        self.assertLessEqual(join.departure - join.arrival, 10)
        self.assertFalse(any(leg.kind == 'fixed' and leg.trip == 'A:2' and leg.departure < join.departure for leg in duty.legs))
        rejected = decoder.decode((self.decision('A:3'),), {100}, audit=True)
        self.assertTrue(rejected.errors)

    def test_search_generates_solo_coupling_and_depot_alternatives(self):
        from stage2.insertion import endings
        n, p, table = scenario()
        decoder = Decoder(n, table, p)
        normal = list(endings(decoder, self.decision('A:1')))
        self.assertTrue(any(e.target.startswith('backup:') for e in normal))
        decoder.allow_solo = True
        alternatives = list(endings(decoder, self.decision('A:1')))
        targets = {e.target for e in alternatives}
        self.assertIn('', targets)
        self.assertIn('depot:1', targets)
        self.assertIn('A:2', targets)

    def test_fixed_donor_returns_then_serves_another_request(self):
        n, p, table = scenario(ordinary=0, drt=2)
        n.passengers = [replace(q, earliest=15, latest=40) if q.id == 101 else q for q in n.passengers]
        first = self.decision('depot:1', 0)
        second = Excursion('A:0/m1', 0, 'backup:A', 2, ((101, True), (101, False)), 'depot', 1, 13)
        result = Decoder(n, table, p).decode((first, second), {100, 101}, audit=True)
        self.assertFalse(result.errors, result.errors)
        duty = result.duties['A:0/m1']
        self.assertEqual(sum(v.kind == 'pickup' for v in duty.visits), 2)
        self.assertEqual(len([item for item in result.metadata['activations'] if item['vehicle'] == duty.vehicle]), 1)
        with tempfile.TemporaryDirectory() as tmp:
            draw(result, n, p, tmp, dpi=40)
            self.assertTrue((Path(tmp) / 'reserve_drt_schedule.png').exists())

    def test_repair_keeps_solo_available_without_a_second_fallback_pass(self):
        import random
        from unittest.mock import patch
        from stage2.constructive_repair import repair
        n, p, table = scenario()
        decoder = Decoder(n, table, p)
        seen = []
        from stage2.initialization import construct

        def build(*args, **kwargs):
            seen.append(args[0].allow_solo)
            return construct(*args, **kwargs)
        with patch('stage2.constructive_repair.construct', side_effect=build):
            result, failure = repair(decoder, (), [100], 0, random.Random(p.seed))
        self.assertIsNone(failure)
        self.assertFalse(result.errors)
        self.assertEqual(seen, [True])
        self.assertTrue(decoder.allow_solo)

    def test_consecutive_orders_without_intermediate_return(self):
        n, p, table = scenario(ordinary=0, drt=2)
        n.passengers = [replace(q, earliest=8, latest=30) if q.id == 101 else q for q in n.passengers]
        events = ((100, True), (100, False), (101, True), (101, False))
        result = Decoder(n, table, p).decode((self.decision('backup:A', events=events),), {100, 101}, audit=True)
        self.assertFalse(result.errors, result.errors)
        duty = result.duties['A:0/m1']
        drop = next(i for i, v in enumerate(duty.visits) if v.kind == 'dropoff' and v.node == n.passengers[0].dest)
        pickup = next(i for i, v in enumerate(duty.visits) if v.kind == 'pickup' and v.node == n.passengers[1].origin)
        self.assertLess(drop, pickup)
        self.assertFalse(any(v.kind in ('depot', 'join', 'stored_join') for v in duty.visits[drop:pickup + 1]))
