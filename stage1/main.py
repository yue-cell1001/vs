import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import monotonic
from common.data import load_network
from .params import Params
from .nsga2 import solve
from .obj import Evaluator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data')
    parser.add_argument('--output', default='results/stage1.json')
    parser.add_argument('--population', type=int, default=Params().population)
    parser.add_argument('--generations', type=int, default=Params().generations)
    parser.add_argument('--evaluation-budget', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--capacity', type=int, default=Params().capacity)
    parser.add_argument('--purchase-cost', type=float, default=Params().purchase_cost)
    parser.add_argument('--operating-cost', type=float, default=Params().operating_cost)
    args = parser.parse_args()
    start = monotonic()
    if args.evaluation_budget < 0:
        parser.error('--evaluation-budget must be nonnegative; 0 disables the limit')
    p = Params(evaluation_budget=args.evaluation_budget or None, population=args.population, generations=args.generations, seed=args.seed, capacity=args.capacity, purchase_cost=args.purchase_cost, operating_cost=args.operating_cost)
    net = load_network(args.data)
    stats = {}
    items = solve(net, p, stats, verify_baseline=True)
    if not items:
        raise SystemExit('No feasible stage 1 schedule')
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    evaluator = Evaluator(net, p)
    for stale in path.parent.glob(f'{path.stem}_reference_*.json'):
        suffix = stale.stem.removeprefix(path.stem + '_reference_')
        if suffix.isdigit() and int(suffix) >= len(items):
            stale.unlink()
    for i, item in enumerate(items):
        detail = evaluator.detail(item['timetable'])
        (path.parent / f'{path.stem}_reference_{i:03}.json').write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding='utf-8')
    from common.convergence import write_csv
    write_csv([('Stage 1', stats['convergence'])], path.with_name(path.stem + '_convergence.csv'))
    stats.update(parameters=asdict(p), elapsed_seconds=monotonic() - start, ordinary_passengers=sum((not q.kind for q in net.passengers)))
    path.with_name(path.stem + '_run.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'{len(items)} schedules -> {path}', flush=True)
if __name__ == '__main__':
    main()
