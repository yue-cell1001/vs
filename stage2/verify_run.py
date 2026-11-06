import argparse
import json
from pathlib import Path
from common.data import load_network
from .params import Params
from .audit import audit_file

def verify(directory, data='data'):
    root = Path(directory)
    run = json.loads((root / 'run.json').read_text(encoding='utf-8'))
    net = load_network(data)
    errors = []
    checked = 0
    stats = run['search']
    failures = run.get('failed_timetables', [])
    if len(stats) + len(failures) != run['total_timetables']:
        errors.append('Unprocessed input timetables')
    for stat in stats:
        folder = root / f"timetable_{stat['timetable'] + 1:02}"
        p = Params(**json.loads((folder / 'run.json').read_text(encoding='utf-8'))['parameters'])
        complete = [row for row in stat['trace'] if row['repair_complete']]
        if len(complete) != stat['complete_feasible_iterations']:
            errors.append('Iteration counter mismatch')
        for row in complete:
            file = folder / 'iterations' / f"iteration_{row['iteration']:03}" / 'solution_000.json'
            if not file.exists():
                errors.append(f'Missing iteration candidate {file}')
                continue
            errors.extend(audit_file(file, net, p))
            raw = json.loads(file.read_text(encoding='utf-8'))
            if raw['metadata']['iteration'] != row['iteration']:
                errors.append('Wrong candidate iteration')
            if any((abs(a - b) > 1e-06 for a, b in zip(raw['objectives'], row['trial_objectives']))):
                errors.append('Trace/objective mismatch')
            checked += 1
        for entry in json.loads((folder / 'pareto.json').read_text(encoding='utf-8')):
            errors.extend(audit_file(folder / (entry['solution'] + '.json'), net, p))
    p = Params(**run['parameters'])
    for entry in json.loads((root / 'pareto.json').read_text(encoding='utf-8')):
        errors.extend(audit_file(root / (entry['solution'] + '.json'), net, p))
    return {'passed': not errors, 'timetables': len(stats), 'expected_timetables': run['total_timetables'], 'initialization_failures': failures, 'complete_repaired_candidates_audited': checked, 'passengers_per_candidate': len(net.passengers), 'errors': errors}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--data', default='data')
    args = parser.parse_args()
    report = verify(args.directory, args.data)
    (Path(args.directory) / 'iteration_verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    if not report['passed']:
        raise SystemExit('\n'.join(report['errors']))
    print(f"PASS: {report['timetables']} feasible branches, {report['complete_repaired_candidates_audited']} complete candidates, {len(report['initialization_failures'])} initialization failures")
if __name__ == '__main__':
    main()
