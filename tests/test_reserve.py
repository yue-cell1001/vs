import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from common.model import Excursion
from stage2.decoder import Decoder
from stage2.constraints import validate
from stage2.obj import evaluate
from stage2.resources import inventory
from stage2.routing import Infeasible
from stage2.plot import draw
from tests.fixtures import scenario as base_scenario

def scenario(*args, **kwargs):
    net, p, table = base_scenario(*args, **kwargs)
    return net, replace(p, depot_initial=5), table

class ReserveTests(unittest.TestCase):

    def test_idle_stock_has_no_purchase_charge(self):
        n, p, table = scenario(ordinary=0, drt=0)
        result = Decoder(n, table, p).decode((), set(), audit=True)
        self.assertFalse(result.errors, result.errors)
        for report in result.metadata['inventory'].values():
            self.assertEqual(report['initial'], 5)
        self.assertEqual(result.components['purchase'], result.metadata['fleet_size'] * p.purchase_cost)
        self.assertLess(result.metadata['fleet_size'], 10)

    def test_standalone_return_and_separate_plot(self):
        n, p, table = scenario()
        p = replace(p, stop_time=20 / 60, operation_time=30 / 60, end=30)
        table = {line: list(range(0, 31, 5)) for line in table}
        n.passengers = [replace(q, earliest=20, latest=30) if q.kind else q for q in n.passengers]
        decision = Excursion('reserve:1/m0', 0, 'depot:1', 0, ((100, True), (100, False)), 'depot', 1, 0)
        result = Decoder(n, table, p).decode((decision,), {100}, audit=True)
        self.assertFalse(result.errors, result.errors)
        activations = [a for a in result.metadata['activations'] if a['origin'] == 'reserve']
        self.assertEqual(len(activations), 1)
        self.assertEqual(result.metadata['inventory'][1]['final'], 5)
        expected = result.objectives
        self.assertEqual(evaluate(result, n, p), expected)
        self.assertGreater(result.components['drt_travel'], 0)
        with tempfile.TemporaryDirectory() as tmp:
            files = draw(result, n, p, tmp, dpi=40)
            self.assertEqual(len(files), len(n.lines) + 1)
            self.assertTrue((Path(tmp) / 'reserve_drt_schedule.png').exists())
            info = json.loads((Path(tmp) / 'plot_info.json').read_text())
            self.assertFalse(info['DRT_offroute_segments_displayed'])

    def test_returned_module_reused_without_new_purchase(self):
        n, p, table = scenario(ordinary=0, drt=2)
        p = replace(p, end=30)
        table = {line: list(range(0, 31, 5)) for line in table}
        n.passengers = [replace(q, earliest=15, latest=40) if q.id == 101 else q for q in n.passengers]
        first = Excursion('reserve:1/m0', 0, 'depot:1', 0, ((100, True), (100, False)), 'depot', 1, 0)
        second = Excursion('reserve:1/m0', 0, 'depot:1', 0, ((101, True), (101, False)), 'depot', 1, 14)
        result = Decoder(n, table, p).decode((first, second), {100, 101}, audit=True)
        self.assertFalse(result.errors, result.errors)
        self.assertEqual(len([a for a in result.metadata['activations'] if a['origin'] == 'reserve']), 1)
        self.assertFalse(validate(result, n, None, p))
        self.assertTrue(any(s['kind'] == 'warehouse_continuation' for s in result.metadata['storage']))

    def test_reserve_can_join_fixed_trip_at_origin(self):
        n, p, table = scenario(ordinary=0, drt=0)
        e = Excursion('reserve:1/m0', 0, 'A:1', 0, (), 'depot', 1, 3)
        result = Decoder(n, table, p).decode((e,), set(), audit=True)
        self.assertFalse(result.errors, result.errors)
        self.assertTrue(any(move['distance'] > 0 for move in result.metadata['rebalance_moves']) or any(leg.kind == 'empty' and leg.distance > 0 for duty in result.duties.values() for leg in duty.legs))

    def test_integer_model_cannot_create_extra_initial_reserve(self):
        n, p, table = scenario(ordinary=0, drt=2)
        p = replace(p, depot_initial=1)
        decisions = tuple(Excursion(f'reserve:1/m{i}', 0, 'depot:1', 0, ((100 + i, True), (100 + i, False)), 'depot', 1, 0) for i in range(2))
        result = Decoder(n, table, p).decode(decisions, {100, 101}, audit=True)
        self.assertTrue(result.errors)

    def test_stock_and_capacity_are_hard_limits(self):
        n, p, _ = scenario()
        self.assertEqual(inventory([], n, p)[1]['final'], 5)
        with self.assertRaises(Infeasible):
            inventory([], n, p, [{'origin': 'reserve', 'node': 1, 'vehicle': i, 'time': 0} for i in range(6)])
        with self.assertRaises(Infeasible):
            inventory([{'node': 1, 'vehicle': i, 'in': 1, 'out': None} for i in range(p.depot_capacity - p.depot_initial + 1)], n, p)
