from pathlib import Path
import random
import unittest
from common.model import Passenger
from common.data import load_network
from stage1.params import Params
from stage1.reference import ReferenceDecoder
from stage1.constraints import repair, violations
from stage1.demand import profiles
from stage1.nsga2 import solve
from stage1.selection import select_candidates
from tests.fixtures import network

class ReferenceTests(unittest.TestCase):

    def parameters(self, **kwargs):
        return Params(start=0, end=15, stop_time=0, **kwargs)

    def table(self, n):
        return {line: [0, 5, 10, 15] for line in n.lines}

    @unittest.skipUnless(all((Path(__file__).resolve().parents[1] / "data" / name).is_file() for name in ("nodes_with_drt.csv", "lines.csv", "edges.csv", "passengers.csv")), "Private Test dataset is not published; provide local CSV files to run this integration check")
    def test_new_data_schema(self):
        net = load_network('data')
        self.assertEqual(sum((not p.kind for p in net.passengers)), 3000)
        self.assertEqual(sum((bool(p.kind) for p in net.passengers)), 300)
        self.assertEqual(len(net.transfer_candidates), 0)
        self.assertEqual(net.line_families['6_up'], net.line_families['6_down'])

    @unittest.skipUnless(all((Path(__file__).resolve().parents[1] / "data" / name).is_file() for name in ("nodes_with_drt.csv", "lines.csv", "edges.csv", "passengers.csv")), "Private Test dataset is not published; provide local CSV files to run this integration check")
    def test_full_input_reference_coverage(self):
        n = load_network('data')
        p = Params()
        table = {line: list(range(480, 601, 15)) for line in n.lines}
        decoder = ReferenceDecoder(n, p)
        self.assertTrue(decoder.decode(table).uncovered)
        table = decoder.allocation.repair(table)
        result = decoder.decode(table)
        self.assertFalse(result.uncovered)
        self.assertEqual(len(result.assignments), 3000)
        self.assertGreater(result.objectives[0], 0)
        self.assertGreater(len(result.trips), 72)
        self.assertTrue(all(t['formation'] <= 4 for t in result.trips.values()))

    def test_boarding_uses_origin_station_arrival(self):
        n = network([Passenger(1, 1, 3, 0, 1.5, 15, 'A')])
        result = ReferenceDecoder(n, self.parameters()).decode(self.table(n))
        self.assertEqual(result.assignments[1]['board'], 6)
        self.assertEqual(result.objectives[1], 4.5)

    def test_peak_load_releases_alighting_capacity(self):
        people = [Passenger(i, 0, 1, 0, 0, 15, 'A') for i in range(6)] + [Passenger(6 + i, 1, 2, 0, 0, 15, 'A') for i in range(6)]
        n = network(people)
        r = ReferenceDecoder(n, self.parameters()).decode(self.table(n))
        self.assertEqual(r.trips['A:0']['peak'], 6)
        self.assertEqual(r.trips['A:0']['formation'], 2)
        self.assertEqual(r.trips['A:0']['loads'][:2], [6, 6])

    def test_empty_services_cost_at_least_one_module(self):
        n = network()
        r = ReferenceDecoder(n, self.parameters()).decode(self.table(n))
        self.assertTrue(all((t['formation'] == 1 for t in r.trips.values())))
        self.assertGreater(r.objectives[0], 0)

    def test_same_line_reuse_and_turnaround(self):
        n = network()
        n.line_families = {'A': 'A', 'Ar': 'A', 'B': 'B', 'Br': 'B'}
        r = ReferenceDecoder(n, self.parameters()).decode(self.table(n))
        self.assertTrue(any((len(m['tasks']) > 1 for m in r.modules)))
        for m in r.modules:
            for a, b in zip(m['tasks'], m['tasks'][1:]):
                self.assertGreaterEqual(r.trips[b]['start'] - r.trips[a]['end'], 1)

    def test_all_reposition_distance_included_in_cost(self):
        n = network()
        n.lines = {'A': n.lines['A']}
        p = self.parameters(depot_initial=0)
        r = ReferenceDecoder(n, p).decode(self.table(n))
        self.assertTrue(any(move['distance'] > 0 for move in r.reposition))
        expected = p.purchase_cost * len(r.modules) + sum(
            p.operating_cost * trip['distance'] * (1 + p.following_ratio * (trip['formation'] - 1))
            for trip in r.trips.values()
        )
        expected += p.operating_cost * sum(move['distance'] for move in r.reposition)
        self.assertAlmostEqual(r.objectives[0], expected)

    def test_capacity_queue_and_hard_formation_limit(self):
        n = network([Passenger(i, 0, 3, 0, 0, 15, 'A') for i in range(30)])
        r = ReferenceDecoder(n, self.parameters()).decode(self.table(n))
        self.assertEqual(r.trips['A:0']['formation'], 4)
        self.assertEqual(r.trips['A:0']['peak'], 24)
        self.assertEqual(r.trips['A:1']['peak'], 6)
        self.assertFalse(r.capacity_risks)
        self.assertFalse(r.uncovered)
        self.assertEqual(r.objectives[1], 30)

    @unittest.skipUnless(all((Path(__file__).resolve().parents[1] / "data" / name).is_file() for name in ("nodes_with_drt.csv", "lines.csv", "edges.csv", "passengers.csv")), "Private Test dataset is not published; provide local CSV files to run this integration check")
    def test_capacity_four_real_repair_and_second_stage_baseline(self):
        from stage1.verification import verify
        n = load_network('data')
        p = Params(capacity=4)
        decoder = ReferenceDecoder(n, p)
        table = {line: list(range(480, 601, 15)) for line in n.lines}
        self.assertTrue(decoder.decode(table).uncovered)
        table = decoder.allocation.repair(table)
        self.assertFalse(violations(table, n, p))
        result = decoder.decode(table)
        self.assertFalse(result.uncovered)
        self.assertEqual(len(result.assignments), 3000)
        self.assertTrue(all(max(t['loads'], default=0) <= 16 for t in result.trips.values()))
        report = verify(n, p, table, result)
        self.assertEqual(report['status'], 'verified_feasible')
        self.assertEqual(report['ordinary_passengers'], 3000)
        self.assertFalse(report['full_drt_feasibility_verified'])

    def test_baseline_timeout_is_unresolved(self):
        from unittest.mock import patch
        from common.model import Solution
        from stage1.verification import verify
        n = network()
        p = self.parameters()
        table = self.table(n)
        reference = ReferenceDecoder(n, p).decode(table)
        with patch('stage1.verification.Decoder.decode', return_value=Solution({}, {}, (), errors=['Joint integer allocation: Time limit reached; no verified incumbent'])):
            self.assertEqual(verify(n, p, table, reference)['status'], 'unresolved')
        with patch('stage1.verification.Decoder.decode', return_value=Solution({}, {}, (), errors=['Joint integer allocation: Infeasible; no verified incumbent'])):
            self.assertEqual(verify(n, p, table, reference)['status'], 'infeasible')

    def test_boarding_during_station_dwell(self):
        n = network([Passenger(1, 0, 3, 0, 0.2, 15, 'A')])
        p = Params(start=0, end=15, stop_time=0.5)
        result = ReferenceDecoder(n, p).decode(self.table(n))
        self.assertEqual(result.assignments[1]['board'], 0.5)
        self.assertAlmostEqual(result.objectives[1], 0.3)

    def test_warehouse_expansion_preserves_first_stage_objectives(self):
        from dataclasses import replace
        n = network()
        p = self.parameters(depot_capacity=10)
        before = ReferenceDecoder(n, p).decode(self.table(n))
        after = ReferenceDecoder(n, replace(p, depot_capacity=20)).decode(self.table(n))
        self.assertEqual(before.objectives, after.objectives)
        self.assertEqual(before.assignments, after.assignments)
        self.assertEqual(before.reposition, after.reposition)

    def test_drt_excluded(self):
        n = network()
        p = self.parameters()
        first = ReferenceDecoder(n, p).decode(self.table(n)).objectives
        n.passengers.append(Passenger(100, 12, 13, 1, 0, 20, ''))
        self.assertEqual(first, ReferenceDecoder(n, p).decode(self.table(n)).objectives)

    def test_uncovered_not_zero_wait(self):
        n = network([Passenger(1, 0, 3, 0, 30, 40, 'A')])
        r = ReferenceDecoder(n, self.parameters()).decode(self.table(n))
        self.assertEqual(r.uncovered, [1])
        self.assertEqual(r.objectives, (float('inf'),) * 2)

    def test_l1_projection_and_coverage(self):
        p = self.parameters()
        self.assertEqual(repair([0, 6, 7, 15], p), [0, 5, 10, 15])
        n = network()
        self.assertTrue(violations({line: [10, 15] for line in n.lines}, n, p))

    def test_no_demand_guidance_midpoint(self):
        n = network()
        p = self.parameters()
        self.assertTrue(all((all((x == 10 for x in v)) for v in profiles(n, p).values())))

    def test_crowding_retains_fifteen_and_extremes(self):
        candidates = [{'timetable': {'A': [0, i, 120]}, 'objectives': (i, 40 - i)} for i in range(40)]
        selected, stats = select_candidates(candidates)
        self.assertEqual(len(selected), 15)
        self.assertEqual(selected[0]['objectives'], (0, 40))
        self.assertEqual(selected[-1]['objectives'], (39, 1))

    def test_evaluation_budget_stops_search_and_exports_evaluated_front(self):
        n = network()
        p = self.parameters(population=4, generations=10000, evaluation_budget=25)
        stats = {}
        items = solve(n, p, stats)
        self.assertTrue(items)
        self.assertEqual(stats['evaluation_calls'], 25)
        self.assertEqual(stats['termination_reason'], 'evaluation_budget')
        self.assertLess(stats['generations_completed'], 10000)
        self.assertLessEqual(stats['unique_evaluations'], 25)
        for item in items:
            self.assertEqual(item['objectives'], ReferenceDecoder(n, p).decode(item['timetable']).objectives)

    def test_small_search_outputs_recomputed_objectives(self):
        n = network([Passenger(1, 1, 3, 0, 1.5, 15, 'A')])
        p = self.parameters(population=4, generations=2)
        stats = {}
        items = solve(n, p, stats)
        self.assertEqual([r['iteration'] for r in stats['convergence']], [0, 1, 2])
        hv = [r['hv'] for r in stats['convergence']]
        self.assertEqual(hv, sorted(hv))
        self.assertEqual(stats['termination_reason'], 'generation_limit')
        decoder = ReferenceDecoder(n, p)
        self.assertTrue(items)
        for item in items:
            self.assertEqual(item['objectives'], decoder.decode(item['timetable']).objectives)
