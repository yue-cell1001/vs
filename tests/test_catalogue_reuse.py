import unittest
from dataclasses import replace
from unittest.mock import patch

from stage2.base import build_base
from stage2.joint import grouped_duties
from stage2.passengers import catalogue, ordinary_itineraries
from common.timetable import make_trips
from tests.fixtures import scenario


class CatalogueReuseTests(unittest.TestCase):
    def test_shared_od_matches_independent_release_time_catalogues(self):
        net, p, table = scenario(ordinary=1, drt=0)
        person = net.passengers[0]
        # Include releases on either side of the existing boarding tolerance.
        releases = [0, 0, 2, 5, 5 + 0.5e-7, 5 + 2e-7, 12]
        net.passengers = [replace(person, id=i, earliest=t) for i, t in enumerate(releases)]
        trips = make_trips(net, table, p)
        groups, _ = grouped_duties(build_base(net, trips, p), set())
        with patch('stage2.passengers.ordinary_itineraries', wraps=ordinary_itineraries) as build:
            columns, requests = catalogue(net, groups, set(), p, trips)
        self.assertEqual(build.call_count, 1)
        self.assertEqual(len(requests), len(set(releases)))
        original = net.passengers
        for count, indices in requests:
            actual = [columns[i] for i in indices]
            ids = set(actual[0].people)
            net.passengers = [person for person in original if person.id in ids]
            expected, expected_requests = catalogue(net, groups, set(), p, trips)
            self.assertEqual(actual, expected)
            self.assertEqual(count, len(net.passengers))
            self.assertEqual(expected_requests, [(count, tuple(range(len(expected))))])
        net.passengers = original

    def test_no_cache_survives_changed_physical_paths(self):
        net, p, table = scenario(ordinary=1, drt=0)
        trips = make_trips(net, table, p)
        groups, _ = grouped_duties(build_base(net, trips, p), set())
        before, _ = catalogue(net, groups, set(), p, trips)
        for duty in groups.values():
            duty.visits[:] = [replace(v, arrival=v.arrival + 1, departure=v.departure + 1) for v in duty.visits]
        after, _ = catalogue(net, groups, set(), p, trips)
        self.assertEqual(len(before), len(after))
        for old, new in zip(before, after):
            self.assertAlmostEqual(new.pickup, old.pickup + 1)
            self.assertAlmostEqual(new.cost, old.cost + 1)


if __name__ == '__main__':
    unittest.main()
