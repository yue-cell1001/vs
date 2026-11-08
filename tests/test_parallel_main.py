import csv
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import run
from stage2.main import main
from stage2.parallel import check_start
from tests.fixtures import scenario


class ParallelMainTests(unittest.TestCase):
    def test_entrypoint_resume_forwards_parallel_options_without_stage_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'stage1.json').write_text('[]')
            argv = ['run', '--output', str(root), '--resume', '--workers', '3', '--start-timetable', '4', '--iterations', '10']
            with patch.object(sys, 'argv', argv), patch('run.subprocess.run') as execute:
                run.main()
            self.assertEqual(execute.call_count, 1)
            command = execute.call_args.args[0]
            self.assertIn('stage2.main', command)
            self.assertEqual(command[command.index('--workers') + 1], '3')
            self.assertEqual(command[command.index('--start-timetable') + 1], '4')
            self.assertIn('--resume', command)

    def test_start_requires_all_earlier_results_to_be_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'Timetable 1 is not completed'):
                check_start(root, 6, 4)
            with self.assertRaisesRegex(ValueError, 'outside'):
                check_start(root, 6, 7)

    def test_resume_from_four_preserves_first_three_and_merges_real_process_results(self):
        n, _, _ = scenario()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / 'data'
            data.mkdir()
            def write_csv(name, fields, rows):
                with (data / name).open('w', newline='', encoding='utf-8') as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
            write_csv('nodes_with_drt.csv', ['id', 'type', 'name', 'x', 'y'], [{'id': node.id, 'type': node.kind, 'name': node.name, 'x': node.x, 'y': node.y} for node in n.nodes.values()])
            write_csv('lines.csv', ['RouteID_Str', 'stop_sequence'], [{'RouteID_Str': line, 'stop_sequence': ';'.join(map(str, stops))} for line, stops in n.lines.items()])
            write_csv('edges.csv', ['from_id', 'to_id', 'distance', 'travel_time'], [{'from_id': a, 'to_id': b, 'distance': distance * 1000, 'travel_time': duration} for (a, b), (distance, duration) in n.edges.items()])
            write_csv('passengers.csv', ['id', 'origin', 'dest', 'type', 'tw_start', 'tw_end', 'line_id'], [{'id': q.id, 'origin': q.origin, 'dest': q.dest, 'type': q.kind, 'tw_start': 480, 'tw_end': 510 if q.kind else 600, 'line_id': q.line} for q in n.passengers])
            tables = root / 'stage1.json'
            tables.write_text(json.dumps([{'timetable': {line: list(range(480, 601, 10)) for line in n.lines}}] * 6))
            output = root / 'stage2'
            args = ['stage2', '--data', str(data), '--timetables', str(tables), '--output', str(output), '--iterations', '1', '--subproblem-seconds', '1', '--no-reports']
            with patch.object(sys, 'argv', args):
                main()
            saved = {str(file.relative_to(output)): hashlib.sha256(file.read_bytes()).hexdigest() for i in range(1, 4) for file in (output / f'timetable_{i:02}').rglob('*') if file.is_file()}
            for i in range(4, 7):
                shutil.rmtree(output / f'timetable_{i:02}')
            from stage2.results_export import export_results
            def checked_export(*values, **kwargs):
                self.assertTrue(all((output / f'timetable_{i:02}/finished.json').exists() for i in range(1, 7)))
                return export_results(*values, **kwargs)
            with patch.object(sys, 'argv', args[:-1] + ['--resume', '--workers', '3', '--start-timetable', '4']), patch('stage2.main.solve', side_effect=AssertionError('Parent must not run searches')), patch('stage2.results_export.export_results', side_effect=checked_export) as reports:
                main()
            self.assertEqual(reports.call_count, 1)
            self.assertEqual(len(list(output.glob('timetable_*/final_figures/*.png'))), 6 * len(n.lines))
            for name, digest in saved.items():
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), digest)
            workers = [json.loads((output / f'timetable_{i:02}/worker_status.json').read_text()) for i in range(4, 7)]
            self.assertGreaterEqual(len({item['pid'] for item in workers}), 2)
            self.assertTrue(all(item['status'] == 'completed' for item in workers))
            result = json.loads((output / 'run.json').read_text())
            self.assertEqual([item['timetable'] for item in result['search']], list(range(6)))
            from stage2.verify_run import verify
            self.assertTrue(verify(output, data)['passed'])
            with patch.object(sys, 'argv', args + ['--resume', '--workers', '3', '--start-timetable', '4']), patch('stage2.parallel.worker', side_effect=AssertionError('Completed results must not run again')):
                main()

    def test_parallel_failure_terminates_workers_and_cancels_queued_jobs(self):
        from unittest.mock import Mock
        from stage2.parallel import run_parallel
        n, p, table = scenario()
        pool = Mock()
        process = Mock()
        process.is_alive.return_value = True
        pool._processes = {1: process}
        future = Mock()
        future.result.side_effect = RuntimeError('Worker failed')
        pool.submit.return_value = future
        with tempfile.TemporaryDirectory() as tmp, patch('stage2.parallel.ProcessPoolExecutor', return_value=pool), patch('stage2.parallel.as_completed', return_value=[future]):
            with self.assertRaisesRegex(RuntimeError, 'Worker failed'):
                run_parallel([{'timetable': table}], p, n, tmp, Path(tmp), (100, 100), (10, 10), 2, False, False)
        process.terminate.assert_called_once()
        self.assertTrue(future.cancel.called)
        pool.shutdown.assert_called_once_with(wait=True, cancel_futures=True)
