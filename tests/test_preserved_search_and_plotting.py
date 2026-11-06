import json
import random
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from stage2.decoder import Decoder
from stage2.initialization import construct
from stage2.constructive_repair import repair
from stage2.routing import Infeasible
from stage2.plot import draw
from tests.fixtures import scenario


class PreservedSearchAndPlottingTests(unittest.TestCase):
    def test_original_repair_retry_order_is_preserved(self):
        n, p, table = scenario(ordinary=1, drt=1)
        decoder = Decoder(n, table, p)
        plan = construct(decoder, 0, random.Random(p.seed), lambda _: None)
        with patch('stage2.constructive_repair.construct', side_effect=[Infeasible('first'), Infeasible('second'), plan]) as build:
            solution, failure = repair(decoder, (), [100], 2, random.Random(p.seed))
        self.assertIsNone(failure)
        self.assertFalse(solution.errors)
        self.assertEqual([c.args[1] for c in build.call_args_list], [2, 0, 1])
        self.assertEqual(solution.metadata['constructive_repair_rule'], 1)

    def test_plot_once_per_timetable_only_after_all_searches_finish(self):
        from stage2.main import main
        n, p, table = scenario(ordinary=1, drt=1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'input.json'
            source.write_text(json.dumps([{'timetable': table}] * 2))
            out = root / 'output'
            def checked_draw(*args, **kwargs):
                for number in (1, 2):
                    record = json.loads((out / f'timetable_{number:02}/finished.json').read_text())
                    self.assertEqual(record['search']['iterations'], 2)
                self.assertFalse(kwargs['include_reserve'])
                return draw(*args, **kwargs)
            argv = ['stage2', '--timetables', str(source), '--output', str(out), '--iterations', '2', '--subproblem-seconds', '1']
            with patch.object(sys, 'argv', argv), patch('stage2.main.Params', side_effect=lambda **kwargs: replace(p, **kwargs)), patch('stage2.main.load_network', return_value=n), patch('stage2.results_export.load_network', return_value=n), patch('stage2.plot.draw', side_effect=checked_draw) as plot:
                main()
            self.assertEqual(plot.call_count, 2)
            self.assertEqual(len(list(out.rglob('*.png'))), 2 * len(n.lines))
            for number in (1, 2):
                self.assertEqual(len(list((out / f'timetable_{number:02}/final_figures').glob('*.png'))), len(n.lines))
            self.assertFalse((out / 'global_pareto_front.png').exists())
            self.assertFalse((out / 'stage2_convergence.png').exists())
            self.assertTrue((out / 'global_pareto_points.csv').exists())
            self.assertTrue((out / 'stage2_convergence.csv').exists())
            stat = json.loads((out / 'timetable_01/finished.json').read_text())['search']
            self.assertGreater(stat['initialization_elapsed_seconds'], 0)
            self.assertGreater(stat['iteration_elapsed_seconds'], 0)

    def test_first_stage_records_convergence_without_drawing(self):
        from stage1.main import main
        from stage1.params import Params
        n, _, _ = scenario(ordinary=0, drt=0)
        base = Params(start=0, end=15)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv = ['stage1', '--output', str(root / 'stage1.json'), '--population', '4', '--generations', '0']
            with patch.object(sys, 'argv', argv), patch('stage1.main.Params', side_effect=lambda **kwargs: replace(base, **kwargs)), patch('stage1.main.load_network', return_value=n), patch('common.convergence.draw', side_effect=AssertionError('No automatic plots before all stage-2 iterations finish')):
                main()
            self.assertTrue((root / 'stage1_convergence.csv').exists())
            self.assertEqual(list(root.rglob('*.png')), [])
