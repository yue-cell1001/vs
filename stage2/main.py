import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from time import monotonic
from common.data import load_network
from common.pareto import Archive
from common.configuration import warehouse_expansion, search_defaults
from .params import Params
from .alns import solve, NoFeasibleSolution
from .audit import restore
from .report import export
from .bounds import objective_bounds
from .scalarization import reference_scales, positive_pair

def _write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data')
    parser.add_argument('--timetables', default='results/stage1.json')
    parser.add_argument('--output', default='results/stage2')
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--subproblem-seconds', type=float, default=Params().subproblem_seconds)
    parser.add_argument('--repair-seconds', type=float)
    parser.add_argument('--initialization', choices=('constructive', 'insertion'), default='constructive')
    parser.add_argument('--repair-mode', choices=('constructive', 'exhaustive'), default='constructive')
    parser.add_argument('--initial-solutions', type=int, default=Params().initial_solutions)
    parser.add_argument('--initial-feasible-target', type=int)
    parser.add_argument('--search-seconds', type=float)
    parser.add_argument('--stall', type=int, default=60)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--capacity', type=int, default=Params().capacity)
    parser.add_argument('--purchase-cost', type=float, default=Params().purchase_cost)
    parser.add_argument('--operating-cost', type=float, default=Params().operating_cost)
    parser.add_argument('--objective-bounds')
    parser.add_argument('--no-reports', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--plot', action='store_true', default=True)
    parser.add_argument('--no-plot', dest='plot', action='store_false')
    parser.add_argument('--require-feasible-iteration', action='store_true')
    args = parser.parse_args()
    if args.retry_failed and not args.resume:
        parser.error('--retry-failed requires --resume')
    start = monotonic()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    p = Params(iterations=args.iterations, subproblem_seconds=args.subproblem_seconds, repair_seconds=args.repair_seconds, seed=args.seed, capacity=args.capacity, purchase_cost=args.purchase_cost, operating_cost=args.operating_cost, initialization=args.initialization, repair_mode=args.repair_mode, initial_solutions=args.initial_solutions, initial_feasible_target=args.initial_feasible_target, search_seconds=args.search_seconds, stall=args.stall)
    net = load_network(args.data)
    candidates = json.loads(Path(args.timetables).read_text(encoding='utf-8'))
    if isinstance(candidates, dict):
        candidates = [candidates]
    digest = hashlib.sha256()
    for file in sorted(Path(args.data).glob('*.csv')):
        digest.update(file.name.encode())
        digest.update(file.read_bytes())
    digest.update(json.dumps(candidates, sort_keys=True).encode())
    fingerprint = digest.hexdigest()
    upper, bound_calculation = objective_bounds(net, candidates, p)
    if args.objective_bounds:
        shared_configuration = json.loads(Path(args.objective_bounds).read_text(encoding='utf-8'))
        shared = shared_configuration['upper']
        if len(shared) != 2 or any((not 0 < local <= fixed for local, fixed in zip(upper, shared))):
            raise ValueError('Shared objective bounds do not cover this case')
        upper = tuple(shared)
    allocation_scales = reference_scales(candidates, upper)
    if args.objective_bounds:
        allocation_scales = positive_pair(shared_configuration.get('allocation_scales', allocation_scales))
    if not args.resume and any(root.iterdir()):
        raise ValueError('Output directory is not empty; use --resume or a new directory')
    if args.resume:
        manifest = search_defaults(json.loads((root / 'configuration.json').read_text(encoding='utf-8')))
        parameters = json.loads(json.dumps(asdict(p)))
        if manifest.get('allocation_scales') != list(allocation_scales):
            raise ValueError('Resume objective scale mismatch')
        if manifest['fingerprint'] != fingerprint:
            raise ValueError('Resume configuration/input mismatch')
        if manifest['parameters'] != parameters:
            if not args.retry_failed or not warehouse_expansion(manifest['parameters'], parameters):
                raise ValueError('Resume configuration/input mismatch')
            if any(root.glob('timetable_*/finished.json')) or any(root.glob('timetable_*/checkpoint.json')):
                raise ValueError('Warehouse capacity changed after a feasible search was saved; use a new output directory')
    elif (root / 'configuration.json').exists():
        raise ValueError('Output exists; use --resume or another directory')
    _write(root / 'configuration.json', {'fingerprint': fingerprint, 'parameters': asdict(p), 'objective_bounds': upper, 'bound_calculation': bound_calculation, 'allocation_scales': allocation_scales})
    previous_elapsed = 0.0
    if args.resume and (root / 'timing.json').exists():
        previous_elapsed = json.loads((root / 'timing.json').read_text(encoding='utf-8')).get('total_elapsed_seconds', 0)
    completed = False
    report_export_seconds = 0.0

    def timing(status):
        elapsed = monotonic() - start
        return {'status': status, 'invocation_elapsed_seconds': elapsed, 'total_elapsed_seconds': previous_elapsed + elapsed, 'total_elapsed_hours': (previous_elapsed + elapsed) / 3600}
    archive = Archive()
    stats = []
    failures = []
    processed = 0
    try:
        for k, candidate in enumerate(candidates):
            folder = root / f'timetable_{k + 1:02}'
            folder.mkdir(exist_ok=True)
            cp = replace(p, seed=p.seed + k)
            recovered = None
            if args.resume and not args.retry_failed and (folder / 'failure.json').exists() and (not (folder / 'checkpoint.json').exists()):
                failures.append(json.loads((folder / 'failure.json').read_text(encoding='utf-8')))
                processed = k + 1
                continue
            saved_path = folder / 'finished.json' if (folder / 'finished.json').exists() else folder / 'checkpoint.json'
            if args.resume and saved_path.exists():
                payload = json.loads(saved_path.read_text(encoding='utf-8'))
                saved = [restore(raw) for raw in payload['solutions']]
                from .constraints import validate
                from .obj import evaluate
                for item in saved:
                    errors = validate(item, net, None, cp)
                    old_objectives = item.objectives
                    actual = evaluate(item, net, cp)
                    if errors or any((abs(a - b) > 1e-06 for a, b in zip(actual, old_objectives))):
                        raise ValueError('Invalid resume snapshot')
                    archive.add(item)
                if saved_path.name == 'finished.json':
                    stat = payload['search']
                    export(saved, net, cp, folder, {'timetable_index': k, 'phase': 'completed', 'search': stat}, tables=False)
                    stats.append({'timetable': k, **stat})
                    processed = k + 1
                    continue
                recovered = {'solutions': saved, 'state': payload['state']}

            def checkpoint(items, phase):
                for item in items:
                    item.metadata['timetable_index'] = k
                destination = folder / 'iterations' / f"iteration_{phase['iteration']:03}" if phase['phase'] == 'iteration_candidate' else folder
                if 'state' in phase:
                    _write(folder / 'checkpoint.json', {'solutions': [item.to_dict() for item in items], 'state': phase['state']})
                export(items, net, cp, destination, {'timetable_index': k, 'checkpoint': {key: value for key, value in phase.items() if key != 'state'}}, tables=False)
                _write(root / 'timing.json', timing('running'))
            print(f'Timetable {k + 1}/{len(candidates)}', flush=True)
            try:
                solutions, stat = solve(net, candidate['timetable'], cp, checkpoint=checkpoint, resume=recovered, global_archive=archive, upper=upper, allocation_scales=allocation_scales)
            except NoFeasibleSolution as error:
                failures.append({'timetable': k, 'error': str(error), **error.diagnostics})
                _write(folder / 'failure.json', failures[-1])
                processed = k + 1
            else:
                export(solutions, net, cp, folder, {'timetable_index': k, 'phase': 'completed', 'search': stat}, tables=False)
                _write(folder / 'finished.json', {'solutions': [item.to_dict() for item in solutions], 'search': stat})
                (folder / 'failure.json').unlink(missing_ok=True)
                stats.append({'timetable': k, **stat})
                processed = k + 1
            _write(root / 'progress.json', {'completed': processed, 'total': len(candidates), 'failures': failures})
            if archive.items:
                export(archive.items, net, p, root, {'search': stats, 'failed_timetables': failures, 'completed_timetables': processed, 'total_timetables': len(candidates)}, tables=False)
        if archive.items:
            export(archive.items, net, p, root, {'search': stats, 'failed_timetables': failures, 'completed_timetables': processed, 'total_timetables': len(candidates)}, tables=False)
            _write(root / 'progress.json', {'completed': processed, 'total': len(candidates), 'failures': failures})
            if not args.no_reports:
                from .results_export import export_results
                export_started = monotonic()
                export_results(root, args.data, plots=args.plot)
                report_export_seconds = monotonic() - export_started
        else:
            raise NoFeasibleSolution('No complete feasible initialization; see per-timetable failure.json')
        if args.require_feasible_iteration and (failures or any((s['complete_feasible_iterations'] < 1 for s in stats))):
            raise RuntimeError('A feasible branch completed no feasible neighbourhood iteration')
        from common.convergence import write_csv
        write_csv([(f"Timetable {s['timetable'] + 1}", s['trace']) for s in stats], root / 'stage2_convergence.csv')
        completed = True
    finally:
        report = timing('completed' if completed else 'interrupted_or_failed')
        report.update(processed_timetables=processed, total_timetables=len(candidates))
        report['report_export_seconds'] = report_export_seconds
        _write(root / 'timing.json', report)
        print(f"Total elapsed: {report['total_elapsed_seconds']:.2f} seconds ({report['total_elapsed_hours']:.3f} hours)", flush=True)
if __name__ == '__main__':
    main()
