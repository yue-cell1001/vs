import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch
import capacity_sensitivity as study


class FirstStageReuseTests(unittest.TestCase):
    def bundle(self, root, settings):
        data, source, target = root / 'data', root / 'old', root / 'new'
        data.mkdir()
        (data / 'data.csv').write_text('unchanged')
        study.write_json(source / 'configuration.json', {'data_fingerprint': study.fingerprint(data), 'stage2_defaults': {'old': True}})
        for q in settings.capacities:
            folder = source / f'capacity_{q}'
            study.write_json(folder / 'stage1.json', [{'timetable': {'A': [480, 600]}, 'objectives': [1, 2]}])
            study.write_json(folder / 'stage1_run.json', {'parameters': asdict(settings.stage1(q)), 'elapsed_seconds': 1})
        return data, source, target

    def test_import_is_independent_of_second_stage_settings_and_does_not_mutate_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = study.Settings()
            data, source, target = self.bundle(Path(tmp), settings)
            before = {p.relative_to(source): p.read_bytes() for p in source.rglob('*.json')}
            changed = replace(settings, iterations=3, subproblem_seconds=5, repair_seconds=10)
            with patch('capacity_sensitivity.violations', return_value=[]):
                study.reuse_first_stage(changed, data, source, target, object())
            for q in settings.capacities:
                folder = target / f'capacity_{q}'
                self.assertTrue(study.read_json(folder / 'case.json')['stage1_complete'])
                self.assertEqual((folder / 'stage1.json').read_bytes(), (source / f'capacity_{q}/stage1.json').read_bytes())
            self.assertEqual(before, {p.relative_to(source): p.read_bytes() for p in source.rglob('*.json')})

    def test_changed_data_is_rejected_before_copying(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = study.Settings()
            data, source, target = self.bundle(Path(tmp), settings)
            (data / 'data.csv').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                study.reuse_first_stage(settings, data, source, target, object())
            self.assertFalse(target.exists())

    def test_changed_first_stage_parameter_is_rejected_before_any_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = study.Settings()
            data, source, target = self.bundle(Path(tmp), settings)
            with patch('capacity_sensitivity.violations', return_value=[]), self.assertRaisesRegex(ValueError, 'parameters'):
                study.reuse_first_stage(replace(settings, seed=99), data, source, target, object())
            self.assertFalse(target.exists())

    def test_missing_capacity_and_illegal_timetable_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = study.Settings()
            data, source, target = self.bundle(Path(tmp), settings)
            (source / 'capacity_8/stage1.json').unlink()
            with patch('capacity_sensitivity.violations', return_value=[]), self.assertRaisesRegex(ValueError, 'Capacity 8'):
                study.reuse_first_stage(settings, data, source, target, object())
            self.assertFalse(target.exists())
            with patch('capacity_sensitivity.violations', return_value=['Invalid timetable']), self.assertRaisesRegex(ValueError, 'illegal'):
                study.reuse_first_stage(settings, data, source, target, object())
