import unittest
from dataclasses import replace
from common.model import Excursion
from stage2.decoder import Decoder
from stage2.initial_returns import endings
from tests.fixtures import scenario


class InitialReturnTests(unittest.TestCase):

    def test_all_return_types_are_ranked_by_operating_cost_without_solo_penalty(self):
        from stage2.continuation_cost import cost
        n, p, table = scenario()
        decoder = Decoder(n, table, p)
        source = decoder.trips['A:0']
        e = Excursion('A:0/m1', 1, '', 0, ((100, True), (100, False)), 'A:0', source.visits[1].node)
        origin = source.visits[1]
        node = n.passengers[-1].dest
        options = list(endings(decoder, decoder.trips, set(), origin, e, node, 5, 25, 1, n.distance))
        costs = [cost(n, p, node, target, j, decoder.trips) for target, j in options]
        self.assertEqual(costs, sorted(costs))
        self.assertTrue(any(target.startswith('backup:') for target, _ in options))
        self.assertTrue(any(target.startswith('depot:') for target, _ in options))
        self.assertTrue(any(target in decoder.trips for target, _ in options))

    def test_cheaper_solo_continuation_can_precede_warehouse_return(self):
        from stage2.continuation_cost import cost
        n, p, table = scenario()
        n.depots = {1: 20}
        n.nodes[1] = replace(n.nodes[1], x=-10000, y=-10000)
        decoder = Decoder(n, table, p)
        origin = decoder.trips['A:0'].visits[0]
        e = Excursion('A:0/m1', 0, '', 0, ((100, True), (100, False)), 'A:0', origin.node)
        trips = {'A:0': decoder.trips['A:0']}
        node = n.passengers[-1].dest
        choices = list(endings(decoder, trips, set(), origin, e, node, 5, 25, 0, n.distance))
        self.assertTrue(choices[0][0].startswith('backup:'))
        self.assertLess(cost(n, p, node, *choices[0], trips), cost(n, p, node, 'depot:1', 0, trips))

    def test_cross_line_returns_require_both_interchanges(self):
        n, p, table = scenario()
        decoder = Decoder(n, table, p)
        source = decoder.trips['A:0']
        origin = source.visits[0]
        self.assertNotIn(origin.node, n.depots)
        e = Excursion('A:0/m1', 0, '', 0, ((100, True), (100, False)), 'A:0', origin.node)
        options = list(endings(decoder, decoder.trips, set(), origin, e, n.passengers[-1].dest, 0, 25, 0, n.distance))
        self.assertTrue(all(target not in decoder.trips or decoder.trips[target].line == origin.line for target, _ in options))
