import argparse
import csv
import json
from math import isfinite
from pathlib import Path
from common.data import load_network
from common.pareto import objective_key
from .audit import audit_file, restore
from .params import Params
from .plot import choose_entry
from .workbook import export_workbook

def nondominated(points):
    groups = {}
    for i, p in enumerate(points):
        key = objective_key((p['F1'], p['F2']))
        if not all((isfinite(v) for v in key)):
            raise ValueError('Nonfinite objectives')
        groups.setdefault(key, []).append(i)
    best = float('inf')
    keep = set()
    for (_, f2), indices in sorted(groups.items()):
        if f2 < best:
            keep.update(indices)
            best = f2
    return keep

def export_timetable(folder, net, plots=True, solution_name=None):
    folder = Path(folder)
    run = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
    if run.get('phase') != 'completed':
        return None
    p = Params(**run['parameters'])
    entries = json.loads((folder / 'pareto.json').read_text(encoding='utf-8'))
    if not entries:
        return None
    solutions = []
    for entry in entries:
        file = folder / (entry['solution'] + '.json')
        errors = audit_file(file, net, p)
        if errors:
            raise ValueError(f'{file}: invalid saved schedule: {errors[:3]}')
        solutions.append(restore(json.loads(file.read_text(encoding='utf-8'))))
    if [e['solution'] for e in entries] != [f'solution_{i:03}' for i in range(len(entries))]:
        raise ValueError('Unexpected solution numbering')
    stat = run.get('search', {})
    number = run['timetable_index'] + 1
    stats = [{**stat, 'timetable': number - 1}]
    workbook = export_workbook(solutions, net, p, folder, stats, solution_name)
    selected, selection = choose_entry(entries, solution_name)
    images = []
    if plots:
        from .plot import draw
        images = draw(solutions[int(selected['solution'].split('_')[-1])], net, p, folder / 'final_figures', include_reserve=False)
        info_file = folder / 'final_figures/plot_info.json'
        info = json.loads(info_file.read_text(encoding='utf-8'))
        info.update(source_solution=selected['solution'] + '.json', selection=selection, objectives=[selected['F1'], selected['F2']])
        info_file.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    return {'timetable': number, 'workbook': str(workbook), 'selected_solution': selected['solution'], 'images': images}

def _front_plot(points, front, root):
    import os
    os.environ.setdefault('MPLCONFIGDIR', str(Path('/tmp') / 'modular_bus_drt_matplotlib') if os.name != 'nt' else str(Path.home() / '.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    tables = sorted({p['timetable'] for p in points})
    colors = plt.get_cmap('tab20')
    for j, number in enumerate(tables):
        rows = [(i, p) for i, p in enumerate(points) if p['timetable'] == number]
        color = colors(j % 20)
        ax.scatter([p['F1'] for _, p in rows], [p['F2'] for _, p in rows], s=12, color=color, alpha=0.15)
        rows = [p for i, p in rows if i in front]
        if rows:
            ax.scatter([p['F1'] for p in rows], [p['F2'] for p in rows], s=35, color=color, label=f'Timetable {number:02}')
    unique = sorted({(points[i]['F1'], points[i]['F2']) for i in front})
    ax.plot([v[0] for v in unique], [v[1] for v in unique], color='black', linewidth=0.8, alpha=0.5, zorder=0)
    ax.set(xlabel='F1: operating cost', ylabel='F2: passenger time (person-min)', title='Global Pareto front of saved feasible schedules')
    ax.grid(alpha=0.2)
    if front:
        ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    file = root / 'global_pareto_front.png'
    fig.savefig(file, dpi=200)
    plt.close(fig)
    return str(file)

def export_results(directory, data='data', plots=True, progress=print, solution_name=None):
    root = Path(directory)
    net = load_network(data)
    reports = []
    points = []
    seen = set()
    skipped = []
    for folder in sorted(root.glob('timetable_*')):
        if not folder.is_dir() or not (folder / 'run.json').exists():
            continue
        run = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
        if run.get('phase') != 'completed':
            skipped.append(folder.name)
            continue
        report = export_timetable(folder, net, plots, solution_name)
        if report is None:
            continue
        reports.append(report)
        progress(f"Exported timetable {report['timetable']}: Excel + {len(report['images'])} schedule images")
        p = Params(**run['parameters'])
        files = list(folder.glob('solution_*.json')) + list((folder / 'iterations').glob('iteration_*/solution_*.json'))
        for file in files:
            if file.name.endswith('_audit.json'):
                continue
            errors = audit_file(file, net, p)
            if errors:
                raise ValueError(f'{file}: invalid schedule: {errors[:3]}')
            raw = json.loads(file.read_text(encoding='utf-8'))
            key = (report['timetable'], *objective_key(raw['objectives']))
            source = str(file.relative_to(root))
            if key in seen:
                continue
            seen.add(key)
            points.append({'timetable': report['timetable'], 'solution_file': source, 'F1': raw['objectives'][0], 'F2': raw['objectives'][1]})
    if not points:
        raise ValueError('No completed feasible timetable results found')
    front = nondominated(points)
    for i, point in enumerate(points):
        point['non_dominated'] = i in front
    with (root / 'global_pareto_points.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    image = None
    expected = None
    if (root / 'progress.json').exists():
        expected = json.loads((root / 'progress.json').read_text(encoding='utf-8')).get('total')
    manifest = {'timetables': reports, 'skipped_unfinished': skipped, 'expected_timetables': expected, 'all_expected_timetables_exported': expected is not None and len(reports) == expected, 'scope': 'Completed timetable archives and all saved feasible iteration candidates; one point per timetable/objective pair. No archive-size truncation. This is an empirical front, not a global optimality proof.', 'point_count': len(points), 'front_count': len(front), 'pareto_image': image}
    (root / 'results_export.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--data', default='data')
    args = parser.parse_args()
    report = export_results(args.directory, args.data)
    print(f"Exported {len(report['timetables'])} timetables; global front: {report['front_count']} points")
    if not report['all_expected_timetables_exported']:
        print('Partial export: some input timetables are unfinished or have no feasible archive; see results_export.json')
if __name__ == '__main__':
    main()
