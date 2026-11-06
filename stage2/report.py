import csv
import json
from dataclasses import asdict
from pathlib import Path
import re

def export(solutions, net, p, directory, stats, *, tables=True):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    def write_json(name, value):
        temporary = directory / (name + '.tmp')
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        temporary.replace(directory / name)
    write_json('run.json', {'parameters': asdict(p), 'passengers': len(net.passengers), 'neighbourhood_truncated': any((v is not None for v in (p.source_limit, p.join_limit, p.insertion_limit, p.pool_limit))), 'subproblem_time_limit_seconds': p.subproblem_seconds, **stats})
    for previous in directory.glob('solution_*'):
        match = re.fullmatch(r'solution_(\d+)(?:_[a-z_]+\.csv|_audit\.json|\.json)', previous.name)
        if match and int(match.group(1)) >= len(solutions):
            previous.unlink()
    summary = []
    for index, s in enumerate(solutions):
        name = f'solution_{index:03}'
        (directory / (name + '_modules.csv')).unlink(missing_ok=True)
        from .schedule import write_tables
        if tables:
            write_tables(s, net, p, directory, name)
        write_json(name + '.json', s.to_dict())
        pooled = sum((len({pid for pid, pickup in e.events if pickup}) > 1 for e in s.excursions))
        transfers = sum((len(service.rides) > 1 for service in s.services.values()))
        cross = sum((a['source_line'] != a['target_line'] and (not a['target'].startswith('depot:')) for a in s.metadata.get('scheduling_actions', ())))
        summary.append({'solution': name, 'timetable_number': s.metadata.get('timetable_index', -1) + 1, 'F1': s.objectives[0], 'F2': s.objectives[1], 'fleet': s.metadata['fleet_size'], 'pooled_excursions': pooled, 'onboard_transfers': transfers, 'cross_line_excursions': cross, 'empty_rebalance_legs': len(s.metadata['rebalance_moves']), 'ordinary_detour_riders': sum((x.detour > 1e-07 for x in s.services.values())), 'allocation_status': s.metadata.get('joint_solver', {}).get('status', 'unknown'), 'primary_gap': s.metadata.get('joint_solver', {}).get('primary_gap'), 'served': len(s.services), 'errors': len(s.errors)})
    write_json('pareto.json', summary)
    if not tables:
        return
    with (directory / 'pareto.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(summary)
