import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import run
from stage1.params import Params
from stage1.reuse import refresh
from tests.fixtures import scenario


class UpdatedModelReuseTests(unittest.TestCase):
    def test_saved_timetable_is_unchanged_while_reference_values_are_recomputed(self):
        n, _, table = scenario(ordinary=2, drt=0)
        p = Params(start=0, end=15, stop_time=0)
        source = [{'timetable': table, 'objectives': [999, 999], 'ordinary_baseline': {'status': 'old'}}]
        original = copy.deepcopy(source)
        updated = refresh(source, n, p)
        self.assertEqual(source, original)
        self.assertEqual(updated[0]['timetable'], table)
        self.assertNotEqual(updated[0]['objectives'], [999, 999])
        self.assertEqual(updated[0]['ordinary_baseline']['status'], 'verified_feasible')

    def test_main_reuses_stage_one_and_forwards_computation_budgets(self):
        n, _, table = scenario(ordinary=2, drt=0)
        first = Params(start=0, end=15, stop_time=0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'old.json'
            text = json.dumps([{'timetable': table, 'objectives': [999, 999]}])
            source.write_text(text)
            output = root / 'new'
            args = ['run', '--reuse-stage1', str(source), '--output', str(output), '--iterations', '200', '--workers', '2', '--initialization-seconds', '300']
            with patch.object(sys, 'argv', args), patch('run.Stage1Params', return_value=first), patch('common.data.load_network', return_value=n), patch('run.subprocess.run') as execute:
                run.main()
            self.assertEqual(source.read_text(), text)
            self.assertEqual(execute.call_count, 1)
            command = execute.call_args.args[0]
            self.assertIn('stage2.main', command)
            self.assertNotIn('stage1.main', command)
            self.assertEqual(command[command.index('--initialization-seconds') + 1], '300.0')
            self.assertEqual(command[command.index('--workers') + 1], '2')
            report = json.loads((output / 'stage1_run.json').read_text())
            self.assertEqual(report['parameters']['max_formation'], 5)
            self.assertEqual(report['parameters']['depot_initial'], 0)
            self.assertEqual(report['generations_completed'], 0)
