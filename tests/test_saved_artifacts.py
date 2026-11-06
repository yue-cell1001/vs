import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from common.model import Excursion
from stage2.audit import audit_file
from stage2.decoder import Decoder
from stage2.report import export
from tests.fixtures import scenario


class SavedArtifactTests(unittest.TestCase):
    def test_reserve_return_audit_deducts_initial_activation(self):
        n, p, table = scenario(ordinary=0, drt=1)
        p = replace(p, depot_capacity=5, depot_initial=5)
        plan = (Excursion('reserve:1/m0', 0, 'depot:1', 0,
                          ((100, True), (100, False)), 'depot', 1, 0),)
        solution = Decoder(n, table, p).decode(plan, {100}, audit=True)
        self.assertFalse(solution.errors, solution.errors)
        self.assertEqual(solution.metadata['inventory'][1]['final'], 5)
        with tempfile.TemporaryDirectory() as tmp:
            export([solution], n, p, tmp, {}, tables=False)
            self.assertEqual(audit_file(Path(tmp) / 'solution_000.json', n, p), [])

    def test_archive_shrink_removes_stale_generated_solutions(self):
        n, p, table = scenario(ordinary=1, drt=0)
        solution = Decoder(n, table, p).decode((), set(), audit=True)
        self.assertFalse(solution.errors)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            export([solution, solution], n, p, root, {}, tables=False)
            self.assertTrue((root / 'solution_001.json').exists())
            (root / 'notes.txt').write_text('keep')
            export([solution], n, p, root, {}, tables=False)
            self.assertFalse((root / 'solution_001.json').exists())
            self.assertEqual((root / 'notes.txt').read_text(), 'keep')
