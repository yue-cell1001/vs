import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import monotonic
from stage1.params import Params as Stage1Params
from stage2.params import Params as Stage2Params

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--output', default='results')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--start-timetable', type=int)
    parser.add_argument('--reuse-stage1', type=Path)
    parser.add_argument('--initialization-seconds', type=float, default=Stage2Params().initialization_seconds)
    parser.add_argument('--repair-seconds', type=float, default=Stage2Params().repair_seconds)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--no-reports', action='store_true', )
    parser.add_argument('--no-plot', action='store_true', )
    parser.add_argument('--data', default='data')
    parser.add_argument('--population', type=int, default=Stage1Params().population)
    parser.add_argument('--generations', type=int, default=Stage1Params().generations)
    parser.add_argument('--evaluation-budget', type=int, default=0)
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--subproblem-seconds', type=float)
    args = parser.parse_args()
    if args.reuse_stage1 is not None and args.resume:
        parser.error('--reuse-stage1 starts a new run and cannot be combined with --resume')
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.start_timetable is not None and not args.resume:
        parser.error('--start-timetable requires --resume')
    if args.retry_failed and not args.resume:
        parser.error('--retry-failed requires --resume')
    if args.verify:
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-v'], check=True)
        return
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume and ((output / 'stage1.json').exists() or (output / 'stage2').exists()):
        raise ValueError('Output exists; choose a new directory or --resume')
    if args.resume and (not (output / 'stage1.json').exists()):
        raise ValueError('Missing first-stage schedules')
    previous = 0.0
    if args.resume:
        if (output / 'total_timing.json').exists():
            previous = json.loads((output / 'total_timing.json').read_text(encoding='utf-8')).get('total_elapsed_seconds', 0)
        else:
            for file in (output / 'stage1_run.json', output / 'stage2/timing.json'):
                if file.exists():
                    report = json.loads(file.read_text(encoding='utf-8'))
                    previous += report.get('total_elapsed_seconds', report.get('elapsed_seconds', 0))
    start = monotonic()
    completed = False
    try:
        if args.reuse_stage1 is not None:
            from dataclasses import asdict
            from common.data import load_network
            from stage1.reuse import refresh
            source = args.reuse_stage1 / 'stage1.json' if args.reuse_stage1.is_dir() else args.reuse_stage1
            saved = json.loads(source.read_text(encoding='utf-8'))
            if isinstance(saved, dict):
                saved = [saved]
            first = Stage1Params(population=args.population, generations=args.generations, evaluation_budget=args.evaluation_budget or None)
            refreshed = refresh(saved, load_network(args.data), first)
            (output / 'stage1.json').write_text(json.dumps(refreshed, ensure_ascii=False, indent=2), encoding='utf-8')
            (output / 'stage1_run.json').write_text(json.dumps({'parameters': asdict(first), 'reused_from': str(source.resolve()), 'reference_recomputed': True, 'generations_completed': 0, 'termination_reason': 'reused_saved_timetables', 'elapsed_seconds': monotonic() - start}, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f'Reused {len(refreshed)} first-stage timetables; reference objectives recomputed', flush=True)
        elif not args.resume:
            subprocess.run([sys.executable, '-m', 'stage1.main', '--data', args.data, '--output', str(output / 'stage1.json'), '--population', str(args.population), '--generations', str(args.generations), '--evaluation-budget', str(args.evaluation_budget)], check=True)
        subprocess.run([sys.executable, '-m', 'stage2.main', '--data', args.data, '--timetables', str(output / 'stage1.json'), '--output', str(output / 'stage2'), '--iterations', str(args.iterations)] + (['--subproblem-seconds', str(args.subproblem_seconds)] if args.subproblem_seconds is not None else []) + (['--resume'] if args.resume else []) + ['--initialization-seconds', str(args.initialization_seconds), '--repair-seconds', str(args.repair_seconds), '--workers', str(args.workers)] + (['--start-timetable', str(args.start_timetable)] if args.start_timetable is not None else []) + (['--retry-failed'] if args.retry_failed else []) + (['--no-reports'] if args.no_reports else []) + (['--no-plot'] if args.no_plot else []), check=True)
        completed = True
    finally:
        elapsed = previous + monotonic() - start
        (output / 'total_timing.json').write_text(json.dumps({'status': 'completed' if completed else 'interrupted_or_failed', 'total_elapsed_seconds': elapsed, 'total_elapsed_hours': elapsed / 3600}, indent=2), encoding='utf-8')
        print(f'Total two-stage elapsed: {elapsed:.2f} seconds ({elapsed / 3600:.3f} hours)', flush=True)
if __name__ == '__main__':
    main()
