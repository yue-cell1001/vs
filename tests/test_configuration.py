import unittest
from common.configuration import warehouse_expansion, search_defaults


class ConfigurationTests(unittest.TestCase):

    def test_only_warehouse_increase_is_compatible(self):
        old = {'parameters': {'depot_capacity': 10, 'depot_initial': 5, 'capacity': 4, 'max_formation': 4}}
        current = {'parameters': {'depot_capacity': 20, 'depot_initial': 5, 'capacity': 4, 'max_formation': 4}}
        self.assertTrue(warehouse_expansion(old, current))
        self.assertFalse(warehouse_expansion(current, old))
        for key, value in [('depot_initial', 10), ('capacity', 5), ('max_formation', 7)]:
            changed = {'parameters': {**current['parameters'], key: value}}
            self.assertFalse(warehouse_expansion(old, changed))
        self.assertFalse(warehouse_expansion(None, current))

    def test_legacy_search_default_does_not_change_operating_parameters(self):
        old = {'initial_checks': 30000, 'initial_backtracks': 3, 'capacity': 6, 'max_formation': 4, 'depot_capacity': 20, 'depot_initial': 5}
        upgraded = search_defaults({'parameters': old})['parameters']
        self.assertEqual(upgraded, {**old, 'repair_checks': 10000})
        self.assertNotIn('repair_checks', old)
        configured = {**old, 'repair_checks': 20000}
        self.assertEqual(search_defaults(configured), configured)
