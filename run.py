import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import monotonic

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--output', default='results')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--no-reports', action='store_true', help='Skip stage-2 Excel/plots; retain JSON results, iteration candidates and resumable checkpoints')
    parser.add_argument('--no-plot', action='store_true', help='Skip stage-2 plots while retaining Excel reports')
    parser.add_argument('--data', default='data')
    parser.add_argument('--population', type=int, default=100)
    parser.add_argument('--generations', type=int, default=300)
    parser.add_argument('--evaluation-budget', type=int, default=0)
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--subproblem-seconds', type=float)
    args = parser.parse_args()
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
        if not args.resume:
            subprocess.run([sys.executable, '-m', 'stage1.main', '--data', args.data, '--output', str(output / 'stage1.json'), '--population', str(args.population), '--generations', str(args.generations), '--evaluation-budget', str(args.evaluation_budget)], check=True)
        subprocess.run([sys.executable, '-m', 'stage2.main', '--data', args.data, '--timetables', str(output / 'stage1.json'), '--output', str(output / 'stage2'), '--iterations', str(args.iterations)] + (['--subproblem-seconds', str(args.subproblem_seconds)] if args.subproblem_seconds is not None else []) + (['--resume'] if args.resume else []) + (['--retry-failed'] if args.retry_failed else []) + (['--no-reports'] if args.no_reports else []) + (['--no-plot'] if args.no_plot else []), check=True)
        completed = True
    finally:
        elapsed = previous + monotonic() - start
        (output / 'total_timing.json').write_text(json.dumps({'status': 'completed' if completed else 'interrupted_or_failed', 'total_elapsed_seconds': elapsed, 'total_elapsed_hours': elapsed / 3600}, indent=2), encoding='utf-8')
        print(f'Total two-stage elapsed: {elapsed:.2f} seconds ({elapsed / 3600:.3f} hours)', flush=True)
if __name__ == '__main__':
    main()
