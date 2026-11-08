import csv
import json
import tempfile
import unittest
from pathlib import Path
from common.model import Excursion
from stage2.decoder import Decoder
from stage2.plot import draw
from tests.fixtures import scenario


class TimetablePlotContentTests(unittest.TestCase):
    def test_direction_plots_include_independent_running_without_drt_markers(self):
        n, p, table = scenario()
        decoder = Decoder(n, table, p)
        decision = Excursion('A:0/m1', 1, 'backup:A', 2, ((100, True), (100, False)), 'A:0', 1)
        solution = decoder.decode((decision,), {100}, audit=True)
        self.assertFalse(solution.errors, solution.errors)
        with tempfile.TemporaryDirectory() as tmp:
            files = draw(solution, n, p, tmp, dpi=40, include_reserve=False)
            self.assertEqual(len(files), len(n.lines))
            info = json.loads((Path(tmp) / 'plot_info.json').read_text())
            self.assertGreater(info['independent_segments'], 0)
            self.assertFalse(info['DRT_position_markers_displayed'])
            self.assertFalse(info['DRT_offroute_segments_displayed'])
            self.assertIsNone(info['reserve_drt_chart'])
            with (Path(tmp) / 'segments.csv').open(encoding='utf-8-sig') as stream:
                segments = list(csv.DictReader(stream))
            solo = [row for row in segments if row['trip'].startswith('backup/')]
            self.assertTrue(solo)
            self.assertTrue(all(row['formation'] == '1' for row in solo))

    def test_one_trip_changes_color_after_split_and_after_join(self):
        n, p, table = scenario(ordinary=7)
        decision = Excursion('A:0/m1', 1, 'A:1', 2, ((100, True), (100, False)), 'A:0', 1)
        solution = Decoder(n, table, p).decode((decision,), {100}, audit=True)
        self.assertFalse(solution.errors, solution.errors)
        with tempfile.TemporaryDirectory() as tmp:
            draw(solution, n, p, tmp, dpi=40, include_reserve=False)
            info = json.loads((Path(tmp) / 'plot_info.json').read_text())
            changes = {(row['trip'], row['before'], row['after']) for row in info['formation_changes']}
            self.assertIn(('A:0', 2, 1), changes)
            self.assertIn(('A:1', 1, 2), changes)
            self.assertEqual(info['formation_color_policy'], 'actual_segment_count')
            self.assertNotEqual(info['formation_colors']['1'], info['formation_colors']['2'])
