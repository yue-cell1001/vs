import csv
import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path
from common.data import load_network
from common.model import Passenger, Service, Solution
from stage1.params import Params as Stage1Params
from stage1.reference import ReferenceDecoder
from stage2.params import Params
from stage2.obj import evaluate
from stage2.learning import Automaton, ACTIONS
from tests.fixtures import network, timetable

ROOT = Path(__file__).resolve().parents[1]


class SettingsTests(unittest.TestCase):
    def test_main_parameters_and_confirmed_reference_load(self):
        p1, p2 = Stage1Params(), Params()
        for p in (p1, p2):
            self.assertEqual((p.capacity, p.max_formation), (6, 4))
            self.assertEqual((p.start, p.end, p.min_headway, p.max_headway), (480, 600, 5, 15))
            self.assertEqual((p.purchase_cost, p.operating_cost, p.following_ratio), (50, .3, .8))
            self.assertEqual((p.depot_initial, p.depot_capacity), (5, 20))
            self.assertAlmostEqual(p.stop_time, 20 / 60)
            self.assertAlmostEqual(p.operation_time, 30 / 60)
        self.assertEqual((p1.population, p1.generations), (64, 200))
        self.assertEqual(p1.reference_load, .8)
        self.assertEqual(p1.candidates, 20)
        self.assertEqual(p2.max_coupling_wait, 10)
        self.assertEqual((p2.learning_rate, p2.exploration, p2.dominance_reward, p2.hv_margin), (.1, .05, .3, .1))
        self.assertEqual(len(ACTIONS), 13)
        self.assertNotIn((4, 0), ACTIONS)
        self.assertNotIn((4, 1), ACTIONS)
        self.assertEqual(Automaton().probabilities, [1 / 13] * 13)

    def test_reference_formation_uses_point_eight_not_point_nine(self):
        n = network([Passenger(i, 0, 3, 0, 0, 100, 'A') for i in range(10)])
        p = Stage1Params(start=0, end=15)
        reference = ReferenceDecoder(n, p).decode(timetable())
        self.assertFalse(reference.uncovered)
        self.assertEqual(reference.trips['A:0']['formation'], 3)

    @unittest.skipUnless((ROOT / "data/source_manifest.json").is_file(), "Data files or source manifest are missing")
    def test_test_zip_data_is_used_without_regeneration(self):
        manifest = json.loads((ROOT / 'data/source_manifest.json').read_text())
        for name, digest in manifest['sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT / 'data' / name).read_bytes()).hexdigest(), digest)
        n = load_network(ROOT / 'data')
        self.assertEqual(Counter(p.kind for p in n.passengers), {0: 3000, 1: 300})
        self.assertEqual(len(n.lines), 8)
        self.assertEqual(len(n.depots), 8)
        self.assertEqual(sum(v.kind in (0, 1) for v in n.nodes.values()), 85)
        self.assertEqual(sum(v.kind == 3 for v in n.nodes.values()), 600)
        for p in n.passengers:
            if p.kind:
                self.assertEqual(p.latest - p.earliest, 30)
                self.assertIn(p.origin_transfer, n.depots)
                self.assertIn(p.destination_transfer, n.depots)
        with (ROOT / 'data/passenger_provenance.csv').open(encoding='utf-8-sig') as stream:
            self.assertEqual(sum(1 for _ in csv.DictReader(stream)), 3000)

    def test_f2_adds_full_detour_without_monetizing_time(self):
        n = network([Passenger(1, 0, 3, 0, 0, 20, 'A'), Passenger(2, 12, 13, 1, 0, 20, '')])
        solution = Solution({}, {1: Service(1, (), 5, 10, 3, 2), 2: Service(2, (), 7, 12)}, ())
        solution.metadata = {'fleet_size': 0, 'rebalance_moves': []}
        self.assertEqual(evaluate(solution, n, Params())[1], 15)
