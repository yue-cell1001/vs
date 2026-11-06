import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import run

class EntryPointTests(unittest.TestCase):

    def test_combined_runtime_and_two_stage_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'run'
            with patch.object(sys, 'argv', ['run', '--output', str(root)]), patch('run.subprocess.run') as execute:
                run.main()
            self.assertEqual(execute.call_count, 2)
            self.assertIn('stage1.main', execute.call_args_list[0].args[0])
            command = execute.call_args_list[0].args[0]
            self.assertEqual(command[command.index('--population') + 1], '64')
            self.assertEqual(command[command.index('--generations') + 1], '200')
            self.assertIn('stage2.main', execute.call_args_list[1].args[0])
            timing = json.loads((root / 'total_timing.json').read_text())
            self.assertEqual(timing['status'], 'completed')
            self.assertGreaterEqual(timing['total_elapsed_seconds'], 0)

    def test_fast_output_flags_reach_second_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(sys, 'argv', ['run', '--output', tmp, '--no-reports', '--no-plot']), patch('run.subprocess.run') as execute:
                run.main()
            first, second = [call.args[0] for call in execute.call_args_list]
            self.assertNotIn('--no-reports', first)
            self.assertIn('--no-reports', second)
            self.assertIn('--no-plot', second)
