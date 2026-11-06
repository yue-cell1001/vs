import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from threading import Barrier, Event, Lock
from unittest.mock import Mock, patch

import capacity_sensitivity as study


class CapacityParallelTests(unittest.TestCase):
    def test_cases_overlap_and_results_keep_capacity_order(self):
        barrier = Barrier(2)
        release = Event()
        lock = Lock()
        active = 0
        maximum = 0
        def work(q):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            barrier.wait(timeout=3)
            if q == 4:
                self.assertTrue(release.wait(timeout=3))
            else:
                release.set()
            with lock:
                active -= 1
            return q
        self.assertEqual(study.map_cases(work, [4, 5], 2, lambda: self.fail('Unexpected abort')), [4, 5])
        self.assertEqual(maximum, 2)

    def test_worker_failure_aborts_and_releases_other_work(self):
        barrier = Barrier(2)
        release = Event()
        def work(q):
            barrier.wait(timeout=3)
            if q == 4:
                raise RuntimeError('Failed case')
            release.wait(timeout=3)
        with self.assertRaisesRegex(RuntimeError, 'Failed case'):
            study.map_cases(work, [4, 5], 2, release.set)
        self.assertTrue(release.is_set())

    def test_cancellation_kills_unresponsive_children_and_blocks_new_starts(self):
        group = study.ProcessGroup()
        process = Mock()
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired('child', 5), 0]
        with patch('capacity_sensitivity.subprocess.Popen', return_value=process):
            self.assertIs(group.start(['child']), process)
        group.cancel()
        process.terminate.assert_called_once()
        process.kill.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            group.start(['child'])

    def test_completed_child_leaves_the_process_group(self):
        group = study.ProcessGroup()
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'stage1.log'
            code = study.execute([sys.executable, '-c', "print('completed')"], log, group)
            self.assertEqual(code, 0)
            self.assertEqual(log.read_text().strip(), 'completed')
        self.assertEqual(group.processes, set())

    def test_worker_count_is_validated_before_starting(self):
        for workers in (0, 6, True, 1.5):
            with self.subTest(workers=workers), self.assertRaises(ValueError):
                study.run(study.Settings(), '.', '.', workers=workers)


if __name__ == '__main__':
    unittest.main()
