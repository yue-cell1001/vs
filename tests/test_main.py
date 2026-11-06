import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from stage2.main import main
from tests.fixtures import scenario

class MainTests(unittest.TestCase):

    def test_capacity_override_shared_bounds_and_fast_export(self):
        from stage2.bounds import objective_bounds
        n, p, table = scenario()
        p = replace(p, capacity=4, purchase_cost=50 * 4 / 6, operating_cost=0.2, iterations=1)
        upper = [x * 2 for x in objective_bounds(n, [{'timetable': table}], p)[0]]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'input.json'
            source.write_text(json.dumps([{'timetable': table}]))
            shared = root / 'upper.json'
            shared.write_text(json.dumps({'upper': upper}))
            args = ['stage2', '--timetables', str(source), '--output', str(root / 'output'), '--capacity', '4', '--purchase-cost', str(50 * 4 / 6), '--operating-cost', '.2', '--iterations', '1', '--subproblem-seconds', '1', '--objective-bounds', str(shared), '--no-reports']
            with patch.object(sys, 'argv', args), patch('stage2.main.Params', side_effect=lambda **kwargs: replace(p, **kwargs)), patch('stage2.main.load_network', return_value=n), patch('stage2.results_export.export_results', side_effect=AssertionError('Fast profile must skip full exports')):
                main()
            result = json.loads((root / 'output/run.json').read_text())
            self.assertEqual(result['parameters']['capacity'], 4)
            self.assertAlmostEqual(result['parameters']['purchase_cost'], 50 * 4 / 6)
            self.assertEqual(result['search'][0]['objective_bounds'], upper)
            self.assertFalse(list((root / 'output').rglob('*.xlsx')))
            self.assertFalse(list((root / 'output').rglob('*.png')))
            self.assertTrue((root / 'output/timetable_01/checkpoint.json').exists())
            self.assertTrue((root / 'output/timetable_01/finished.json').exists())
            self.assertTrue((root / 'output/timetable_01/iterations').exists())
            with patch.object(sys, 'argv', args + ['--resume']), patch('stage2.main.Params', side_effect=lambda **kwargs: replace(p, **kwargs)), patch('stage2.main.load_network', return_value=n), patch('stage2.main.solve', side_effect=AssertionError('Fast-output results must resume without rerunning')):
                main()

    def test_complete_run_and_resume_skip_finished_branch(self):
        n, p, table = scenario()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_file = root / 'input.json'
            input_file.write_text(json.dumps([{'timetable': table}]))
            args = ['stage2', '--timetables', str(input_file), '--output', str(root / 'output'), '--iterations', '1', '--subproblem-seconds', '1', '--plot']

            def factory(**kwargs):
                return replace(p, **kwargs)
            with patch.object(sys, 'argv', args), patch('stage2.main.Params', side_effect=factory), patch('stage2.main.load_network', return_value=n), patch('stage2.results_export.load_network', return_value=n):
                main()
            folder = root / 'output'
            timing = json.loads((folder / 'timing.json').read_text())
            self.assertEqual(timing['status'], 'completed')
            self.assertGreater(timing['total_elapsed_seconds'], 0)
            self.assertFalse((folder / 'stage2_convergence.png').exists())
            self.assertEqual(len(list(folder.rglob('*.png'))), len(n.lines))
            self.assertTrue((folder / 'stage2_convergence.csv').exists())
            self.assertTrue((folder / 'timetable_01/checkpoint.json').exists())
            self.assertTrue((folder / 'timetable_01/finished.json').exists())
            old_manifest = json.loads((folder / 'configuration.json').read_text())
            old_manifest['parameters'].pop('repair_checks', None)
            (folder / 'configuration.json').write_text(json.dumps(old_manifest))
            with patch.object(sys, 'argv', args + ['--resume']), patch('stage2.main.Params', side_effect=factory), patch('stage2.main.load_network', return_value=n), patch('stage2.results_export.load_network', return_value=n), patch('stage2.main.solve', side_effect=AssertionError('Should not solve finished branch')):
                main()
            later = json.loads((folder / 'timing.json').read_text())
            self.assertGreater(later['total_elapsed_seconds'], timing['total_elapsed_seconds'])
            manifest = json.loads((folder / 'configuration.json').read_text())
            manifest['parameters']['depot_capacity'] = 10
            (folder / 'configuration.json').write_text(json.dumps(manifest))
            with patch.object(sys, 'argv', args + ['--resume', '--retry-failed']), patch('stage2.main.Params', side_effect=factory), patch('stage2.main.load_network', return_value=n), patch('stage2.main.solve', side_effect=AssertionError('Saved feasible search must not be mixed across capacity settings')):
                with self.assertRaisesRegex(ValueError, 'Warehouse capacity changed'):
                    main()


    def test_retry_failed_reconstructs_without_replacing_first_stage(self):
        from stage2.alns import NoFeasibleSolution
        n, p, table = scenario()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'input.json'
            original = json.dumps([{'timetable': table}])
            source.write_text(original)
            output = root / 'output'
            args = ['stage2', '--timetables', str(source), '--output', str(output), '--iterations', '0', '--subproblem-seconds', '1', '--no-reports']
            factory = lambda **kwargs: replace(p, **kwargs)
            with patch.object(sys, 'argv', args), patch('stage2.main.Params', side_effect=factory), patch('stage2.main.load_network', return_value=n), patch('stage2.main.solve', side_effect=NoFeasibleSolution('Previous initialization failed')):
                with self.assertRaises(NoFeasibleSolution):
                    main()
            failed = output / 'timetable_01/failure.json'
            self.assertTrue(failed.exists())
            with patch.object(sys, 'argv', args + ['--resume']), patch('stage2.main.Params', side_effect=factory), patch('stage2.main.load_network', return_value=n), patch('stage2.main.solve', side_effect=AssertionError('Ordinary resume preserves recorded failure')):
                with self.assertRaises(NoFeasibleSolution):
                    main()
            configuration = json.loads((output / 'configuration.json').read_text())
            configuration['parameters']['depot_capacity'] = 10
            (output / 'configuration.json').write_text(json.dumps(configuration))
            with patch.object(sys, 'argv', args + ['--resume', '--retry-failed']), patch('stage2.main.Params', side_effect=factory), patch('stage2.main.load_network', return_value=n):
                main()
            self.assertFalse(failed.exists())
            self.assertTrue((output / 'timetable_01/finished.json').exists())
            self.assertEqual(source.read_text(), original)
