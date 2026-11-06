import unittest
from dataclasses import replace
from common.model import Excursion
from stage2.decoder import Decoder
from stage2.initial_capacity import CapacityGuard
from stage2.routing import Infeasible
from tests.fixtures import scenario


class InitialConstructionTests(unittest.TestCase):

    def test_capacity_guard_protects_ordinary_riders_before_insertion(self):
        n, p, _ = scenario(ordinary=2, drt=1)
        n.lines = {'A': n.lines['A'], 'Ar': n.lines['Ar']}
        p = replace(p, capacity=1, max_formation=2)
        decoder = Decoder(n, {'A': [0], 'Ar': [0]}, p)
        fixed = Excursion('A:0/m1', 1, 'depot:1', 0, ((100, True), (100, False)), 'A:0', 1)
        context = decoder.context((fixed,))
        guard = CapacityGuard(n, p)
        with self.assertRaisesRegex(Infeasible, 'Ordinary capacity protection'):
            guard.check(context.trips, context.duties, (fixed,))
        reserve = Excursion('reserve:1/m0', 0, 'depot:1', 0, ((100, True), (100, False)), 'depot', 1, 0)
        context = decoder.context((reserve,))
        guard.check(context.trips, context.duties, (reserve,))
        result = decoder.decode((reserve,), {100}, audit=True)
        self.assertFalse(result.errors, result.errors)

    def test_fast_draft_defers_actual_boarding_feasibility_to_final_joint_model(self):
        n, p, _ = scenario(ordinary=7)
        n.lines = {'A': n.lines['A'], 'Ar': n.lines['Ar']}
        n.passengers = [replace(q, origin=1) if not q.kind else q for q in n.passengers]
        p = replace(p, max_formation=2)
        decoder = Decoder(n, {'A': [0], 'Ar': [0]}, p)
        decision = Excursion('A:0/m1', 0, 'backup:A', 2, ((100, True), (100, False)), 'A:0', 0)
        context = decoder.context((decision,))
        CapacityGuard(n, p).check(context.trips, context.duties, (decision,))
        result = decoder.decode((decision,), {100}, audit=True)
        self.assertTrue(result.errors)
        self.assertTrue(any('Infeasible' in error for error in result.errors), result.errors)

    def test_uploaded_construction_leaves_ordinary_assignment_to_final_joint_model(self):
        import random
        from stage2.initialization import construct
        n, p, table = scenario(ordinary=7, drt=2)
        decoder = Decoder(n, table, p)
        from unittest.mock import patch
        with patch.object(decoder, 'decode', side_effect=AssertionError('Construction must not preallocate ordinary modules')):
            plan = construct(decoder, 0, random.Random(p.seed), progress=lambda _: None)
        self.assertIsNone(decoder.optimizer.initial_hint)
        result = decoder.decode(plan, {100, 101}, audit=True)
        self.assertFalse(result.errors, result.errors)
        self.assertFalse(result.metadata['joint_solver']['initial_reservation_seed'])
        self.assertEqual(len(result.services), 9)
        self.assertEqual({pid for e in plan for pid, pickup in e.events if pickup}, {100, 101})

    def test_unused_warehouse_templates_do_not_break_the_baseline_seed(self):
        from unittest.mock import patch
        from common.integer import solve_integer, _verified
        n, p, table = scenario(ordinary=1, drt=0)
        decoder = Decoder(n, table, p)
        with patch('stage2.joint.solve_integer', wraps=solve_integer) as solver:
            result = decoder.decode((), set(), audit=True)
        self.assertFalse(result.errors, result.errors)
        args = solver.call_args_list[0].args
        self.assertIsNotNone(args[7])
        self.assertIsNotNone(_verified(args[7], args[1], args[2], args[3], args[4], args[5]))

    def test_pickup_can_shift_within_its_window_to_meet_coupling(self):
        n, p, table = scenario()
        decoder = Decoder(n, table, p)
        decision = Excursion('A:0/m1', 1, 'A:3', 2, ((100, True), (100, False)), 'A:0', 1)
        result = decoder.decode((decision,), {100}, audit=True)
        self.assertFalse(result.errors, result.errors)
        self.assertGreater(result.services[100].pickup, 3)
        duty = result.duties[decision.source]
        join = next(v for v in duty.visits if v.trip == 'A:3' and v.kind == 'join')
        self.assertLessEqual(join.departure - p.operation_time - join.arrival, p.max_coupling_wait + 1e-07)

    def test_capacity_guard_keeps_legal_ordinary_detours(self):
        n, p, table = scenario(ordinary=7)
        p = replace(p, max_formation=2)
        decoder = Decoder(n, table, p)
        decision = Excursion('A:0/m1', 1, 'backup:A', 2, ((100, True), (100, False)), 'A:0', 1)
        context = decoder.context((decision,))
        CapacityGuard(n, p).check(context.trips, context.duties, (decision,))
        result = decoder.decode((decision,), {100}, audit=True)
        self.assertFalse(result.errors, result.errors)

    def test_draft_warehouse_capacity_includes_initial_reserves(self):
        from stage2.initial_inventory import check_inventory
        n, p, table = scenario(ordinary=0, drt=2)
        p = replace(p, depot_initial=5, depot_capacity=6)
        decoder = Decoder(n, table, p)
        decisions = (
            Excursion('A:0/m1', 1, 'depot:1', 0, ((100, True), (100, False)), 'A:0', 1),
            Excursion('A:0/m2', 1, 'depot:1', 0, ((101, True), (101, False)), 'A:0', 1),
        )
        first = decoder.context(decisions[:1])
        report = check_inventory(first.duties, decisions[:1], n, p)
        self.assertEqual(report[1]['final'], 6)
        both = decoder.context(decisions)
        with self.assertRaisesRegex(Infeasible, 'Warehouse 1 inventory 7'):
            check_inventory(both.duties, decisions, n, p)

    def test_draft_warehouse_dispatch_releases_its_initial_slot(self):
        from stage2.initial_inventory import check_inventory
        n, p, table = scenario(ordinary=0, drt=1)
        p = replace(p, depot_initial=1, depot_capacity=1)
        decoder = Decoder(n, table, p)
        decision = Excursion('reserve:1/m0', 0, 'depot:1', 0, ((100, True), (100, False)), 'depot', 1, 0)
        context = decoder.context((decision,))
        report = check_inventory(context.duties, (decision,), n, p)
        self.assertEqual(report[1]['final'], 1)
        self.assertEqual(report[1]['peak'], 1)
