import copy
import random
import unittest
from dataclasses import replace
from common.model import Excursion
from common.timetable import make_trips
from stage1.params import Params as Stage1Params
from stage2.params import Params
from stage2.decoder import Decoder
from stage2.constraints import validate
from stage2.initialization import construct
from stage2.routing import Infeasible
from stage2.service_limits import completion_limits, direct_drt_time, duty_in_horizon
from stage2.synchronization import service_times
from tests.fixtures import scenario


class ServiceLimitsTests(unittest.TestCase):
    def test_confirmed_defaults_apply_to_both_stages(self):
        for p in (Stage1Params(), Params()):
            self.assertEqual(p.max_formation, 5)
            self.assertEqual(p.depot_initial, 0)
            self.assertEqual(p.depot_capacity, 20)
        self.assertEqual(Params().max_drt_detour, 15)

    def test_completion_limits_come_from_each_last_fixed_trip(self):
        n, p, table = scenario(ordinary=0, drt=0)
        table['A'][-1] = 17
        trips = make_trips(n, table, p)
        limits = completion_limits(trips)
        self.assertEqual(limits['A'], trips['A:3'].visits[-1].arrival)
        self.assertNotEqual(limits['A'], limits['Ar'])
        self.assertGreater(limits['A'], max(table['A']))

    def test_ride_limit_can_delay_pickup_instead_of_rejecting_a_feasible_pool(self):
        n, p, _ = scenario(ordinary=0, drt=2)
        p = replace(p, max_drt_detour=15)
        people = {q.id: replace(q, earliest=20, latest=25) if q.id == 101 else replace(q, earliest=0, latest=20) for q in n.passengers}
        events = ((100, True), (101, True), (100, False), (101, False))
        answer = service_times(n, people, events, people[100].origin, 0, None, None, p)
        self.assertGreater(answer[0], 0)
        self.assertLessEqual(answer[2] - answer[0], direct_drt_time(n, people[100], p) + 15 + 1e-7)

    def test_impossible_pooled_ride_is_rejected_before_integer_allocation(self):
        n, p, _ = scenario(ordinary=0, drt=2)
        n.nodes[202] = replace(n.nodes[202], x=30000, y=30000)
        n.nodes[203] = replace(n.nodes[203], x=31000, y=30000)
        people = {q.id: replace(q, latest=200) for q in n.passengers}
        events = ((100, True), (101, True), (101, False), (100, False))
        with self.assertRaisesRegex(Infeasible, 'extra ride limit'):
            service_times(n, people, events, people[100].origin, 0, None, None, p)

    def test_independent_validator_rejects_excess_extra_ride(self):
        n, p, table = scenario()
        decoder = Decoder(n, table, p)
        plan = construct(decoder, 0, random.Random(p.seed), progress=lambda _: None)
        solution = decoder.decode(plan, {100}, audit=True)
        self.assertFalse(solution.errors, solution.errors)
        bad = copy.deepcopy(solution)
        bad.services[100] = replace(bad.services[100], dropoff=bad.services[100].dropoff + 100, detour=100)
        self.assertTrue(any('DRT extra ride time' in error for error in validate(bad, n, None, p)))

    def test_scheduled_return_is_used_when_available(self):
        n, p, table = scenario()
        decoder = Decoder(n, table, p)
        plan = construct(decoder, 0, random.Random(p.seed), progress=lambda _: None)
        self.assertTrue(all(e.target in decoder.nominal_trips for e in plan))

    def test_independent_return_is_available_when_no_later_trip_can_couple(self):
        n, p, table = scenario(ordinary=0)
        n.nodes[12] = replace(n.nodes[12], x=0, y=0)
        n.nodes[13] = replace(n.nodes[13], x=500, y=0)
        n.passengers = [replace(q, latest=.1) for q in n.passengers]
        table = {line: [0] for line in table}
        decoder = Decoder(n, table, p)
        plan = construct(decoder, 0, random.Random(p.seed), progress=lambda _: None)
        self.assertTrue(all(e.target.startswith('backup:') for e in plan))
        solution = decoder.decode(plan, {100}, audit=True)
        self.assertFalse(solution.errors, solution.errors)
        limits = completion_limits(decoder.trips)
        self.assertTrue(all(duty_in_horizon(duty, limits) for duty in solution.duties.values()))

    def test_new_task_selects_departure_near_request_instead_of_first_trip(self):
        n, p, table = scenario(ordinary=0)
        n.nodes[12] = replace(n.nodes[12], x=0, y=0)
        n.nodes[13] = replace(n.nodes[13], x=500, y=0)
        n.passengers = [replace(q, earliest=10, latest=11) for q in n.passengers]
        decoder = Decoder(n, table, p)
        plan = construct(decoder, 0, random.Random(p.seed), progress=lambda _: None)
        self.assertEqual(plan[0].split_trip, 'A:2')

    def test_late_task_cannot_extend_the_fixed_trip_horizon(self):
        n, p, table = scenario()
        decoder = Decoder(n, table, p)
        late = Excursion('A:0/m1', 1, 'backup:A', 2, ((100, True), (100, False)), 'A:0', 1)
        n.passengers = [replace(q, earliest=30, latest=40) if q.kind else q for q in n.passengers]
        result = Decoder(n, table, p).decode((late,), {100}, audit=True)
        self.assertTrue(result.errors)

    def test_zero_initial_stock_cannot_supply_a_reserved_warehouse_departure(self):
        n, p, table = scenario()
        self.assertEqual(p.depot_initial, 0)
        decision = Excursion('reserve:1/m0', 0, 'depot:1', 0, ((100, True), (100, False)), 'depot', 1, 0)
        result = Decoder(n, table, p).decode((decision,), {100}, audit=True)
        self.assertTrue(result.errors)
