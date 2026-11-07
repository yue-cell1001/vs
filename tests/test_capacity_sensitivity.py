import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import capacity_sensitivity as study


def point(x, y):
    return {'F1': x, 'F2': y, 'fleet': 1, 'waiting_minutes': y, 'detour_minutes': 0,
            'purchase_cost': x / 2, 'travel_cost': x / 2}


def cases(fronts):
    return [{'capacity': q, 'budget_complete': True, 'front': rows, 'status': 'completed',
             'planned_timetables': 1, 'feasible_timetables': 1, 'iterations_per_timetable': [20]}
            for q, rows in zip(study.CAPACITIES, fronts)]


class CapacitySensitivityTests(unittest.TestCase):
    def test_defaults_and_exact_linear_prices(self):
        settings = study.Settings()
        self.assertEqual(settings.capacities, (4, 5, 6, 7, 8))
        self.assertEqual(settings.iterations, 10)
        self.assertEqual((settings.population, settings.generations), (64, 200))
        for q in settings.capacities:
            purchase, operating = settings.costs(q)
            self.assertAlmostEqual(purchase, 50 * q / 6)
            self.assertAlmostEqual(operating, .3 * q / 6)
            self.assertEqual(settings.stage1(q).max_formation, 4)
            self.assertEqual(settings.stage1(q).candidates, 20)
            self.assertEqual(settings.stage1(q).reference_load, .8)
            self.assertEqual(settings.stage2(q).max_formation, 4)
            self.assertEqual(settings.stage2(q).initial_solutions, 3)
            self.assertEqual(settings.stage2(q).repair_mode, 'constructive')
        self.assertEqual(settings.costs(4)[1], .2)
        self.assertEqual(settings.costs(8)[1], .4)

    def test_child_commands_keep_algorithms_and_skip_all_child_plots(self):
        settings = study.Settings(subproblem_seconds=30)
        folder = Path('/tmp/example')
        command = study.stage2_command(settings, 5, Path('/tmp/data'), folder, Path('/tmp/bounds.json'), True)
        self.assertEqual(command[command.index('--iterations') + 1], '10')
        self.assertEqual(command[command.index('--capacity') + 1], '5')
        self.assertIn('--no-reports', command)
        self.assertIn('--resume', command)
        self.assertIn('--retry-failed', command)
        self.assertNotIn('--initial-solutions', command)
        self.assertNotIn('--initial-feasible-target', command)
        self.assertNotIn('--repair-mode', command)

    def test_nondominated_front_deduplicates_and_retains_tradeoffs(self):
        rows = [point(1, 5), point(2, 4), point(3, 3), point(2, 5), point(3, 3)]
        self.assertEqual([(p['F1'], p['F2']) for p in study.nondominated_rows(rows)], [(1, 5), (2, 4), (3, 3)])
        with self.assertRaises(ValueError):
            study.nondominated_rows([point(float('nan'), 1)])

    def test_observed_front_dominance_can_identify_a_capacity(self):
        result = study.analyze(cases([[point(10 + q, 10 + q)] if q != 6 else [point(6, 6)] for q in study.CAPACITIES]))
        self.assertEqual(result['recommended_capacity'], 6)
        self.assertTrue(result['preference_free_observed_best'])

    def test_crossing_fronts_have_only_a_conditional_hv_recommendation(self):
        data = cases([[point(1, 10)], [point(6, 6)], [point(10, 1)], [point(11, 11)], [point(11, 11)]])
        result = study.analyze(data)
        self.assertEqual(result['recommended_capacity'], 5)
        self.assertFalse(result['preference_free_observed_best'])
        self.assertEqual(result['observed_front_covering_capacities'], [])
        self.assertAlmostEqual(result['cases'][1]['hypervolume'], .36)

    def test_missing_or_unfinished_capacities_never_get_final_recommendation(self):
        data = cases([[point(q, 10 - q)] for q in study.CAPACITIES])
        self.assertIsNone(study.analyze(data[:-1])['recommended_capacity'])
        data[-1]['budget_complete'] = False
        self.assertIsNone(study.analyze(data)['recommended_capacity'])
        self.assertIsNone(study.analyze(cases([[]] * 5))['recommended_capacity'])

    def test_identical_fronts_do_not_force_unique_best(self):
        result = study.analyze(cases([[point(1, 1)]] * 5))
        self.assertIsNone(result['recommended_capacity'])
        self.assertEqual(len(result['highest_hv_capacities']), 5)

    def test_report_has_only_the_requested_front_chart_and_numeric_objectives(self):
        data = cases([[point(q * 100, 1000 / q), point(q * 110, 900 / q)] for q in study.CAPACITIES])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = study.write_report(root, data, study.Settings(), [10000, 10000])
            self.assertTrue((root / 'capacity_pareto_front.png').exists())
            self.assertTrue((root / 'capacity_pareto_front.pdf').exists())
            self.assertEqual(len(list(root.glob('*.png'))), 1)
            self.assertEqual((root / 'pareto_fronts.csv').read_text(encoding='utf-8-sig').splitlines()[0], 'capacity,F1,F2')
            self.assertEqual(json.loads((root / 'analysis.json').read_text())['recommended_capacity'], result['recommended_capacity'])
            self.assertIn('不预设成本必然先降后升', (root / 'analysis.md').read_text())

    def test_changed_input_cannot_resume_previous_study(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'configuration.json').write_text('{}')
            with patch('capacity_sensitivity.load_network', return_value=object()):
                with self.assertRaisesRegex(ValueError, 'Data/model/search settings changed'):
                    study.run(study.Settings(), root, root, resume=True)


if __name__ == '__main__':
    unittest.main()
