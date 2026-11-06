import random
import unittest
from copy import deepcopy
from dataclasses import replace
from common.model import Excursion, Passenger, Service, Ride, Solution, Duty, Visit, Leg
from common.pareto import Archive
from stage2.decoder import Decoder
from stage2.initialization import construct
from stage2.alns import solve
from stage2.obj import evaluate
from stage2.constraints import validate
from stage2.timing import shifted_candidates, validate_branch
from tests.fixtures import scenario, network

class PhysicalTests(unittest.TestCase):

    def test_original_constructor_routes_all_requests_without_joint_solves(self):
        from unittest.mock import patch
        n, p, table = scenario(ordinary=1, drt=2)
        decoder = Decoder(n, table, p)
        with patch.object(decoder, 'decode', side_effect=AssertionError('Original construction does not solve allocations')):
            plan = construct(decoder, 0, random.Random(p.seed), progress=lambda _: None)
        self.assertEqual({pid for e in plan for pid, pickup in e.events if pickup}, {100, 101})
        self.assertTrue(all(not e.target for e in plan))
        self.assertFalse(decoder.decode(plan, {100, 101}, audit=True).errors)

    def test_original_constructor_does_not_invoke_added_draft_guard(self):
        from unittest.mock import patch
        from stage2.initial_capacity import CapacityGuard
        n, p, table = scenario()
        decoder = Decoder(n, table, p)
        with patch.object(CapacityGuard, 'check', side_effect=AssertionError('Do not change original candidate screening')):
            plan = construct(decoder, 0, random.Random(p.seed), progress=lambda _: None)
        self.assertFalse(decoder.decode(plan, {100}, audit=True).errors)

    def test_original_initialization_uses_all_configured_rules(self):
        from unittest.mock import patch
        n, p, table = scenario()
        with patch('stage2.initialization.construct', wraps=construct) as build:
            items, stats = solve(n, table, replace(p, iterations=0), progress=lambda _: None)
        self.assertEqual([c.args[1] for c in build.call_args_list], [0, 1, 2])
        self.assertTrue(items)
        self.assertFalse(stats['initial_failures'])

    def test_original_explicit_feasible_target_stops_initialization(self):
        from unittest.mock import patch
        n, p, table = scenario()
        with patch('stage2.initialization.construct', wraps=construct) as build:
            items, stats = solve(n, table, replace(p, iterations=0, initial_feasible_target=1), progress=lambda _: None)
        self.assertEqual(build.call_count, 1)
        self.assertFalse(validate(items[0], n, None, p))

    def test_original_constructor_orders_requests_by_deadline(self):
        from unittest.mock import patch
        from stage2.initialization import sequence
        n, p, table = scenario(ordinary=0, drt=2)
        n.passengers = [replace(q, latest=10) if q.id == 101 else q for q in n.passengers]
        decoder = Decoder(n, table, p)
        seen = []
        def observed(*args):
            for pid, pickup in args[2]:
                if pickup and pid not in seen:
                    seen.append(pid)
            return sequence(*args)
        with patch('stage2.initialization.sequence', side_effect=observed):
            plan = construct(decoder, 0, random.Random(p.seed), progress=lambda _: None)
        self.assertEqual(seen, [101, 100])
        self.assertFalse(decoder.decode(plan, {100, 101}, audit=True).errors)

    def test_initialization_and_complete_iterations(self):
        n, p, table = scenario()
        snapshots = []
        archive, stats = solve(n, table, p, progress=lambda _: None, checkpoint=lambda items, phase: snapshots.append((list(items), deepcopy(phase))))
        self.assertEqual(stats['iterations'], 2)
        self.assertEqual(stats['complete_feasible_iterations'], 2)
        self.assertEqual(len([phase for _, phase in snapshots if phase['phase'] == 'iteration_candidate']), 2)
        for s in archive:
            self.assertFalse(validate(s, n, None, p))
            saved = s.objectives
            self.assertEqual(evaluate(s, n, p), saved)
            self.assertEqual(s.objectives[1], s.components['normal_wait_minutes'] + s.components['drt_wait_minutes'] + s.components['normal_detour_minutes'])
            self.assertTrue(all(e.target or e.join > e.split for e in s.excursions))
            self.assertEqual(set(s.services), {1, 100})

    def test_ordinary_stations_allowed_on_same_direction(self):
        n, p, table = scenario()
        n.depots = {}
        s, _ = solve(n, table, replace(p, iterations=0), progress=lambda _: None)
        self.assertTrue(s)
        self.assertTrue(any((e.split_node not in n.depots for e in s[0].excursions)))

    def test_original_implicit_continuation_remains_supported(self):
        n, p, table = scenario()
        d = Decoder(n, table, p)
        for target in ('',):
            s = d.decode((Excursion('A:0/m1', 0, target, 1, ((100, True), (100, False)), 'A:0', 0),), {100}, audit=True)
            self.assertFalse(s.errors, s.errors)

    def test_cross_line_split_requires_interchange(self):
        n, p, table = scenario(ordinary=0, drt=0)
        d = Decoder(n, table, p)
        bad = d.decode((Excursion('A:0/m1', 0, 'B:1', 1, (), 'A:0', 0),), set(), audit=True)
        self.assertTrue(bad.errors)
        good = d.decode((Excursion('A:0/m1', 1, 'B:1', 1, (), 'A:0', 1),), set(), audit=True)
        self.assertFalse(good.errors, good.errors)

    def test_ordinary_cannot_change_module_or_line(self):
        n, p, table = scenario()
        items, _ = solve(n, table, replace(p, iterations=0), progress=lambda _: None)
        s = deepcopy(items[0])
        original = s.services[1]
        s.services[1] = replace(original, rides=original.rides + original.rides)
        self.assertTrue(any(('transfer count' in e for e in validate(s, n, None, p))))

    def test_resumed_random_and_learning_states_match(self):
        n, p, table = scenario()
        full, fullstat = solve(n, table, p, progress=lambda _: None)
        saved = {}

        class Stop(Exception):
            pass

        def checkpoint(items, phase):
            if phase['phase'] == 'search' and phase['iteration'] == 1:
                saved['solutions'] = deepcopy(items)
                saved['state'] = deepcopy(phase['state'])
                raise Stop()
        with self.assertRaises(Stop):
            solve(n, table, p, progress=lambda _: None, checkpoint=checkpoint)
        continued, stat = solve(n, table, p, progress=lambda _: None, resume=saved)
        self.assertEqual({s.signature() for s in full}, {s.signature() for s in continued})
        self.assertEqual(fullstat['learner'], stat['learner'])

    def test_micro_adjustment_keeps_branch_bounds(self):
        n, p, table = scenario()
        table = {line: [0, 7.5, 15] for line in table}
        tables = list(shifted_candidates(n, table, table, p, random.Random(2)))
        self.assertGreater(len(tables), 1)
        self.assertTrue(all((validate_branch(t, table, p) for t in tables)))
        changed = deepcopy(table)
        changed['A'][1] += 3
        self.assertFalse(validate_branch(changed, table, p))

    def test_cross_line_drt_transfer_preserves_ordinary_module(self):
        n, p, table = scenario(ordinary=5, drt=1)
        n.nodes[12] = replace(n.nodes[12], x=1000, y=1100)
        n.nodes[13] = replace(n.nodes[13], x=1750, y=750)
        n.nodes[8] = replace(n.nodes[6], id=8, x=500, y=1500)
        n.lines['B'] = (4, 1, 5, 6, 8)
        n.lines['Br'] = (8, 6, 5, 1, 4)
        n.edges[6, 8] = (0.5, 1.0)
        n.edges[8, 6] = (0.5, 1.0)
        d = Decoder(n, table, p)
        pickup = Excursion('A:0/m1', 1, 'B:1', 1, ((100, True),), 'A:0', 1)
        delivery = Excursion('B:1/m2', 2, 'B:2', 3, ((100, False),), 'B:1', 5)
        s = d.decode((pickup, delivery), {100}, audit=True)
        self.assertFalse(s.errors, s.errors)
        self.assertEqual(len(s.services[100].rides), 2)
        for pid, service in s.services.items():
            if pid != 100:
                self.assertEqual(len(service.rides), 1)
        self.assertFalse(validate(s, n, None, p))

    def test_ordinary_cannot_ride_cross_dispatch_before_alighting(self):
        n, p, table = scenario(ordinary=12, drt=1)
        n.passengers = [replace(q, dest=2) if not q.kind else q for q in n.passengers]
        n.lines['B'] = (4, 1, 2, 5, 6)
        n.edges[2, 5] = (0.7071067811865476, 1.4142135623730951)
        d = Decoder(n, table, p)
        e = Excursion('A:0/m1', 1, 'B:1', 2, ((100, True), (100, False)), 'A:0', 1)
        s = d.decode((e,), {100}, audit=True)
        self.assertFalse(s.errors, s.errors)
        for pid, service in s.services.items():
            if pid != 100:
                r = service.rides[0]
                self.assertFalse(r.duty == e.source and r.alight > 1)

    def test_standard_stop_and_operation_times(self):
        n, p, table = scenario(ordinary=3, drt=1)
        p = replace(p, stop_time=20 / 60, operation_time=0.5, iterations=1)
        items, stat = solve(n, table, p, progress=lambda _: None)
        self.assertEqual(stat['complete_feasible_iterations'], 1)
        self.assertTrue(all((not validate(s, n, None, p) for s in items)))

    def test_pickup_wait_is_at_virtual_stop(self):
        n, p, table = scenario()
        n.passengers = [replace(q, earliest=3) if q.kind else q for q in n.passengers]
        d = Decoder(n, table, p)
        e = Excursion('A:0/m1', 0, 'A:1', 2, ((100, True), (100, False)), 'A:0', 0)
        s = d.decode((e,), {100}, audit=True)
        self.assertFalse(s.errors, s.errors)
        self.assertEqual(s.services[100].pickup, 3)
        self.assertTrue(any((v.kind == 'pickup_wait' for v in s.duties[e.source].visits)))

    def test_resource_neighbour_generates_cross_line_coupling(self):
        from stage2.rebalancing import resource_neighbours
        from itertools import islice
        n, p, table = scenario(ordinary=0, drt=0)
        d = Decoder(n, table, p)
        s = d.decode((), set(), audit=True)
        neighbours = list(islice(resource_neighbours(d, s), 30))
        self.assertTrue(neighbours)
        self.assertTrue(any((c[-1].target.startswith('B:') for c in neighbours)))
        self.assertTrue(any((not d.decode(c, set(), audit=True).errors for c in neighbours)))

    def test_reposition_minimizes_distance_subject_to_time(self):
        from common.roads import RoadPaths
        n, p, _ = scenario(ordinary=0, drt=0)
        n.edges = {(0, 1): (10.0, 1.0), (0, 2): (1.0, 2.0), (2, 1): (1.0, 2.0)}
        roads = RoadPaths(n, p)
        self.assertEqual(roads.reposition(0, 1, 3)[:2], (1.0, 10.0))
        self.assertEqual(roads.reposition(0, 1, 5)[:2], (4.0, 2.0))
        self.assertIsNone(roads.reposition(0, 1, 0.5))

    def test_following_cost_ratio_point_eight(self):
        n, p, _ = scenario(ordinary=0, drt=0)
        leg = Leg(0, 1, 0, 1, 0.5, 'fixed', 'A:0', 'A')
        s = Solution({str(i): Duty(str(i), [Visit(0, 0, 0), Visit(1, 1, 1)], [leg], i) for i in range(2)}, {}, ())
        s.metadata = {'fleet_size': 2, 'rebalance_moves': []}
        evaluate(s, n, p)
        self.assertAlmostEqual(s.components['fixed_travel'], 0.3 * 0.5 * 1.8)

    def test_module_numbering_canonical(self):
        n, p, table = scenario()
        items, _ = solve(n, table, replace(p, iterations=0), progress=lambda _: None)
        first = items[0]
        renamed = deepcopy(first)
        for duty in renamed.duties.values():
            duty.vehicle += 100
        self.assertEqual(first.signature(), renamed.signature())
        archive = Archive()
        self.assertTrue(archive.add(first))
        self.assertFalse(archive.add(renamed))

    def test_pooling_and_four_module_capacity(self):
        n, p, table = scenario(ordinary=20, drt=2)
        decoder = Decoder(n, table, p)
        plan = construct(decoder, 0, random.Random(p.seed), progress=lambda _: None)
        items = [decoder.decode(plan, {q.id for q in n.passengers if q.kind}, audit=True)]
        self.assertTrue(items)
        self.assertTrue(any((any((len({pid for pid, _ in e.events}) > 1 for e in s.excursions)) for s in items)))
        for s in items:
            self.assertEqual(len(s.services), 22)
            self.assertFalse(validate(s, n, None, p))
