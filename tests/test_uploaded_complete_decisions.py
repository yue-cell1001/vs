import unittest
from dataclasses import replace
from copy import deepcopy
from common.model import Passenger, Node, Excursion, Duty, Visit, Leg
from common.timetable import make_trips, Trip
from tests.legacy_fixture_adapter import Params
from stage2.decoder import Decoder
from stage2.joint import JointOptimizer
from stage2.obj import evaluate
from stage2.constraints import validate
from .fixtures import network, timetable

class CompleteDecisionTests(unittest.TestCase):

    def test_one_module_repeatedly_detaches_and_rejoins_on_same_chain(self):
        people = [Passenger(1, 12, 13, 1, 0, 20, ''), Passenger(2, 14, 15, 1, 9, 30, '')]
        net = network(people)
        d = Decoder(net, timetable(), Params())
        first = Excursion('A:0/m1', 0, 'A:1', 1, ((1, True), (1, False)))
        second = Excursion('A:0/m1', 0, 'backup:A', 2, ((2, True), (2, False)), 'A:1', 1)
        result = d.decode((first, second), {1, 2}, audit=True)
        self.assertFalse(result.errors, result.errors)
        self.assertEqual(result.services[1].rides[0].duty, result.services[2].rides[0].duty)
        self.assertEqual(result.services[1].rides[0].duty, 'A:0/m1')
        self.assertGreater(result.services[2].pickup, result.services[1].dropoff)

    def test_nine_sequential_orders_are_not_limited_to_four_or_eight_total(self):
        net = network()
        events = []
        for i in range(9):
            a, b = (100 + 2 * i, 101 + 2 * i)
            net.nodes[a] = Node(a, 3, str(a), 100 + 40 * i, 100)
            net.nodes[b] = Node(b, 3, str(b), 120 + 40 * i, 100)
            net.passengers.append(Passenger(i + 1, a, b, 1, 0, 100, ''))
            events.extend(((i + 1, True), (i + 1, False)))
        d = Decoder(net, timetable(), Params(capacity=1))
        result = d.decode((Excursion('A:0/m1', 0, 'backup:A', 2, tuple(events)),), set(range(1, 10)), audit=True)
        self.assertFalse(result.errors, result.errors)
        self.assertEqual(len(result.services), 9)
        self.assertEqual(len({s.rides[0].duty for s in result.services.values()}), 1)

    def test_operation_dwell_is_conditional_and_origins_follow_equation_24(self):
        net = network()
        p = Params()
        table = timetable()
        base = make_trips(net, table, p)
        changed = make_trips(net, table, p, {('A:0', 1)})
        trip = changed['A:0']
        reference = base['A:0']
        self.assertEqual(trip.visits[0].arrival, table['A'][0])
        self.assertAlmostEqual(trip.visits[0].departure - trip.visits[0].arrival, p.stop_time)
        self.assertAlmostEqual(trip.visits[1].departure - trip.visits[1].arrival, p.stop_time + p.operation_time)
        self.assertAlmostEqual(trip.visits[2].arrival - reference.visits[2].arrival, p.operation_time)
        self.assertEqual(changed['B:0'], base['B:0'])

    def test_joint_assignment_resolves_the_fifo_capacity_counterexample(self):
        people = [Passenger(1, 0, 2, 0, 0, 20, 'A'), Passenger(2, 0, 3, 0, 0, 20, 'A')]
        net = network(people)
        p = Params(capacity=1)
        trip = make_trips(net, {'A': [0]}, p)['A:0']
        a = Duty('A', list(trip.visits), list(trip.legs))
        b = Duty('B', list(trip.visits[:3]), list(trip.legs[:2]))
        b.visits[-1] = replace(b.visits[-1], kind='depot')
        s = JointOptimizer(net, p).solve({'A': a, 'B': b}, {'A:0': trip}, (), set())
        self.assertEqual(s.metadata['joint_solver']['status'], 'scalar_optimal')
        self.assertEqual(s.services[1].rides[0].duty, 'B')
        self.assertEqual(s.services[2].rides[0].duty, 'A')

    def test_global_rebalancing_preserves_a_module_needed_by_a_later_line(self):
        net = network()
        p = Params()
        net.edges.update({(0, 1): (0.25, 0.5), (3, 2): (0.25, 0.5), (1, 4): (0.05, 0.1), (2, 1): (1, 2), (1, 0): (0.5, 1)})

        def duty(key, a, b, start, travel, end_kind='fixed'):
            visits = [Visit(a, start, start, key, 'A'), Visit(b, start + travel, start + travel, key, 'A', end_kind)]
            leg = Leg(a, b, start, start + travel, net.edges[a, b][0], 'fixed', key, 'A')
            return Duty(key, visits, [leg])
        units = {'I': duty('I', 0, 1, 0, 0.5, 'depot'), 'J': duty('J', 3, 2, 0, 0.5, 'depot'), 'B': duty('B', 4, 1, 3.2, 1), 'C': duty('C', 0, 1, 3.3, 0.5)}
        net.edges[4, 1] = (0.5, 1)
        trips = {key: Trip(key, 'A', tuple(d.visits), tuple(d.legs)) for key, d in units.items()}
        s = JointOptimizer(net, p).solve(units, trips, (), set())
        self.assertEqual(s.metadata['joint_solver']['status'], 'scalar_optimal')
        self.assertEqual(s.metadata['fleet_size'], 2)
        self.assertEqual(s.duties['B'].vehicle, s.duties['J'].vehicle)
        self.assertEqual(s.duties['C'].vehicle, s.duties['I'].vehicle)

    def test_empty_module_cross_line_join_supplies_intermediate_demand(self):
        people = [Passenger(i, 1, 5, 0, 6, 20, 'B') for i in range(1, 10)]
        d = Decoder(network(people), timetable(), Params())
        e = Excursion('A:0/m1', 1, 'B:1', 1, ())
        s = d.decode((e,), set(), audit=True)
        self.assertFalse(s.errors, s.errors)
        self.assertTrue(any((service.rides[0].duty == e.source for service in s.services.values())))
        self.assertEqual(s.duties[e.source].visits[-1].line, 'B')

    def test_independent_audit_detects_omitted_stock_and_false_operation_masks(self):
        d = Decoder(network(), timetable(), Params())
        a = Excursion('A:3/m1', 1, 'depot:2', 0, ())
        s = d.decode((a,), set(), audit=True)
        self.assertFalse(s.errors, s.errors)
        bad = deepcopy(s)
        bad.metadata['storage'] = []
        self.assertTrue(any(('Warehouse records' in e for e in validate(bad, d.net, d.trips, d.p))))
        bad = deepcopy(s)
        bad.metadata['operations'] = []
        self.assertTrue(any(('dwell mask' in e for e in validate(bad, d.net, d.trips, d.p))))

    def test_allocation_uses_one_symmetric_integer_solve(self):
        from unittest.mock import patch
        from common.integer import solve_integer
        d = Decoder(network(), timetable(), Params())
        with patch('stage2.joint.solve_integer', wraps=solve_integer) as solver:
            solution = d.decode((), set(), audit=True)
        self.assertFalse(solution.errors, solution.errors)
        self.assertEqual(solver.call_count, 1)
        numeric = solution.metadata['joint_solver']
        self.assertEqual(numeric['optimization_method'], 'augmented_weighted_chebyshev')
        self.assertEqual(numeric['tradeoff_weights'], [.5, .5])
        self.assertEqual(numeric['status'], 'scalar_optimal')
        self.assertNotIn('primary_status', numeric)
        self.assertNotIn('secondary_status', numeric)
