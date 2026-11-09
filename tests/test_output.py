import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree as ET
from zipfile import ZipFile
from stage2.alns import solve
from stage2.report import export
from stage2.results_export import export_results
from stage2.verify_run import verify
from stage2.params import Params
from stage2.plot import draw
from tests.fixtures import scenario

class OutputTests(unittest.TestCase):

    def test_complete_export_and_independent_audit(self):
        n, p, table = scenario()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / 'timetable_01'

            def checkpoint(items, phase):
                for s in items:
                    s.metadata['timetable_index'] = 0
                if phase['phase'] == 'iteration_candidate':
                    export(items, n, p, folder / 'iterations' / f"iteration_{phase['iteration']:03}", {}, tables=False)
            items, stat = solve(n, table, p, progress=lambda _: None, checkpoint=checkpoint)
            export(items, n, p, folder, {'phase': 'completed', 'timetable_index': 0, 'search': stat}, tables=False)
            export(items, n, p, root, {'search': [{'timetable': 0, **stat}], 'failed_timetables': [], 'total_timetables': 1, 'completed_timetables': 1}, tables=False)
            (root / 'progress.json').write_text(json.dumps({'completed': 1, 'total': 1}))
            with patch('stage2.results_export.load_network', return_value=n):
                report = export_results(root, progress=lambda _: None)
            self.assertTrue(report['all_expected_timetables_exported'])
            self.assertEqual(len(list(folder.rglob('*.xlsx'))), 1)
            self.assertTrue(all((folder / 'final_figures' / f'{line}.png').exists() for line in n.lines))
            self.assertEqual(len(list(root.rglob('*.png'))), len(n.lines))
            self.assertFalse((root / 'global_pareto_front.png').exists())
            self.assertFalse(list(root.rglob('README*')))
            info = json.loads((folder / 'final_figures/plot_info.json').read_text())
            self.assertFalse(info['DRT_offroute_segments_displayed'])
            self.assertEqual(len(info['formation_colors']), p.max_formation)
            with ZipFile(folder / 'final_dispatch.xlsx') as z:
                for name in z.namelist():
                    ET.fromstring(z.read(name))
            with patch('stage2.verify_run.load_network', return_value=n):
                self.assertTrue(verify(root)['passed'])
            (folder / 'iterations/iteration_001/solution_000.json').unlink()
            with patch('stage2.verify_run.load_network', return_value=n):
                self.assertFalse(verify(root)['passed'])
