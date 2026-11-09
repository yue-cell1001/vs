import argparse
import csv
import json
import os
from math import hypot, isfinite
from pathlib import Path
from .audit import restore, audit_file
from .params import Params
from common.data import load_network

def line_order(line):
    base, _, direction = line.rpartition('_')
    return (int(base) if base.isdigit() else 10 ** 9, base, 0 if direction == 'up' else 1, line)

def segments(solution, net, p):
    arcs = {}
    for duty in solution.duties.values():
        for leg in duty.legs:
            if leg.kind not in ('fixed', 'solo'):
                continue
            key = (leg.line, leg.trip, leg.start, leg.end, round(leg.departure, 7), round(leg.arrival, 7))
            arcs.setdefault(key, set()).add(duty.vehicle)
    rows = []
    for (line, trip, a, b, departure, arrival), vehicles in sorted(arcs.items()):
        stops = net.lines[line]
        if not 1 <= len(vehicles) <= p.max_formation:
            raise ValueError('Invalid formation')
        rows.append({'line': line, 'trip': trip, 'start': a, 'end': b, 'start_order': stops.index(a) + 1, 'end_order': stops.index(b) + 1, 'departure': departure, 'arrival': arrival, 'formation': len(vehicles), 'vehicle_ids': ';'.join(map(str, sorted(vehicles)))})
    return rows

def draw(solution, net, p, directory, dpi=200, include_reserve=True):
    os.environ.setdefault('MPLCONFIGDIR', str(Path('/tmp') / 'modular_bus_drt_matplotlib') if os.name != 'nt' else str(Path.home() / '.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.collections import LineCollection
    from matplotlib.ticker import FuncFormatter, MultipleLocator
    from matplotlib import font_manager
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in ('Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC') if f in fonts), None)
    if font:
        plt.rcParams['font.sans-serif'] = [font]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = segments(solution, net, p)
    outputs = []
    if not rows:
        raise ValueError('No active fixed-line segments')
    colors = {1: '#FFD000', 2: '#66CC55', 3: '#1FA6A6', 4: '#6F168A', 5: '#D55E00'}

    def clock(value, _):
        second = round(value * 60)
        return f'{second // 3600:02}:{second // 60 % 60:02}'
    for line in sorted(net.lines, key=line_order):
        fig, ax = plt.subplots(figsize=(12, 3.8))
        incoming = {}
        outgoing = {}
        paths = []
        palette = []
        data = [r for r in rows if r['line'] == line]
        for row in data:
            paths.append([(row['departure'], row['start_order']), (row['arrival'], row['end_order'])])
            palette.append(colors[row['formation']])
            incoming[row['trip'], row['end']] = (row['arrival'], row['end_order'])
            outgoing[row['trip'], row['start']] = (row['departure'], row['start_order'], row['formation'])
        for key, (arrival, station) in incoming.items():
            if key in outgoing:
                departure, _, n = outgoing[key]
                if departure < arrival - 1e-06:
                    raise ValueError('Non-monotone dwell')
                paths.append([(arrival, station), (departure, station)])
                palette.append(colors[n])
        for row in data:
            if row['trip'].startswith(('solo/', 'backup/')) and (row['trip'], row['start']) not in incoming:
                visit = next(v for duty in solution.duties.values() for v in duty.visits if v.trip == row['trip'] and v.node == row['start'])
                paths.append([(visit.arrival, row['start_order']), (row['departure'], row['start_order'])])
                palette.append(colors[1])
            if row['start_order'] == 1:
                if row['trip'].startswith(('solo/', 'backup/')):
                    start = row['departure']
                else:
                    i = int(row['trip'].rsplit(':', 1)[1])
                    start = solution.metadata['timetable'][line][i]
                paths.append([(start, 1), (row['departure'], 1)])
                palette.append(colors[row['formation']])
        ax.add_collection(LineCollection(paths, colors=palette, linewidths=0.65))
        start = min((r['departure'] for r in data), default=p.start)
        end = max((r['arrival'] for r in data), default=p.end)
        start = min(start, solution.metadata.get('timetable', {}).get(line, [start])[0])
        ax.set_xlim(start - 1, end + 1)
        ax.set_ylim(0.8, len(net.lines[line]) + 0.5)
        ax.set_yticks(sorted({1, *range(2, len(net.lines[line]) + 1, 2), len(net.lines[line])}))
        ax.yaxis.set_minor_locator(MultipleLocator(1))
        ax.xaxis.set_major_locator(MultipleLocator(next((step for step in (5, 10, 15, 30, 60, 120) if (end - start) / step <= 9), 180)))
        ax.xaxis.set_major_formatter(FuncFormatter(clock))
        ax.grid(axis='y', which='both', color='#888888', ls=':', lw=0.45, alpha=0.8)
        ax.tick_params(axis='both', labelsize=9)
        base, _, direction = line.rpartition('_')
        ax.set_xlabel('时间' if font else 'Time')
        ax.set_ylabel('站序' if font else 'Station')
        ax.set_title(f'{base}路' + ('上行' if direction == 'up' else '下行') + '时刻表及车辆编组' if font else f'Route {base} {direction}: timetable and formation', fontsize=11)
        ax.legend(handles=[Line2D([0], [0], color=colors[n], label=f'{n} MU' + ('s' if n > 1 else ''), lw=1.5) for n in range(1, p.max_formation + 1)], loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=p.max_formation, frameon=False, fontsize=9)
        fig.tight_layout()
        file = directory / f'{line}.png'
        fig.savefig(file, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        outputs.append(str(file))
    with (directory / 'segments.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    changes = []
    grouped = {}
    for row in rows:
        grouped.setdefault((row['line'], row['trip']), []).append(row)
    for (line, trip), parts in grouped.items():
        parts.sort(key=lambda row: (row['start_order'], row['departure']))
        for before, after in zip(parts, parts[1:]):
            if before['end'] == after['start'] and before['formation'] != after['formation']:
                changes.append({'line': line, 'trip': trip, 'station': after['start_order'],
                                'time': after['departure'], 'before': before['formation'], 'after': after['formation']})
    reserve_chart = draw_reserve_drt(solution, directory, dpi) if include_reserve else None
    if reserve_chart:
        outputs.append(reserve_chart)
    (directory / 'plot_info.json').write_text(json.dumps({'files': outputs, 'fixed_segments': len(rows), 'DRT_offroute_segments_displayed': False, 'DRT_position_markers_displayed': False, 'independent_segments': sum(r['trip'].startswith(('solo/', 'backup/')) for r in rows), 'reserve_drt_chart': reserve_chart, 'station_axis': 'fixed stop ordinal', 'formation_colors': colors, 'formation_color_policy': 'actual_segment_count', 'formation_changes': changes, 'source': 'actual physical module fixed and independent continuation legs'}, indent=2), encoding='utf-8')
    return outputs

def choose_entry(entries, name=None):
    if not entries:
        raise ValueError('Empty archive')
    if any((not isfinite(entry[key]) for entry in entries for key in ('F1', 'F2'))):
        raise ValueError('Nonfinite archive objective')
    if name is not None:
        found = next((e for e in entries if e['solution'] == name), None)
        if found is None:
            raise ValueError('Unknown solution')
        return (found, {'method': 'explicit_solution'})
    low = [min((e[f'F{j}'] for e in entries)) for j in (1, 2)]
    high = [max((e[f'F{j}'] for e in entries)) for j in (1, 2)]
    distance = [hypot(*[(e[f'F{j + 1}'] - low[j]) / (high[j] - low[j]) if high[j] > low[j] else 0 for j in (0, 1)]) for e in entries]
    index = min(range(len(entries)), key=lambda i: (distance[i], entries[i]['F1'], entries[i]['F2'], entries[i]['solution']))
    return (entries[index], {'method': 'normalized_distance_to_ideal', 'distance': distance[index]})

def plot_file(filename, data='data', output=None, dpi=200, selection=None):
    file = Path(filename)
    run = json.loads((file.parent / 'run.json').read_text(encoding='utf-8'))
    p = Params(**run['parameters'])
    net = load_network(data)
    errors = audit_file(file, net, p)
    if errors:
        raise ValueError(str(errors[:3]))
    target = Path(output) if output else file.parent / 'final_figures'
    return draw(restore(json.loads(file.read_text(encoding='utf-8'))), net, p, target, dpi, include_reserve=False)

def plot_archive(directory, data='data', output=None, dpi=200, solution=None):
    root = Path(directory)
    entry, selection = choose_entry(json.loads((root / 'pareto.json').read_text(encoding='utf-8')), solution)
    return (entry['solution'], plot_file(root / (entry['solution'] + '.json'), data, output, dpi, selection))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('result')
    parser.add_argument('--data', default='data')
    parser.add_argument('--solution')
    parser.add_argument('--output')
    args = parser.parse_args()
    root = Path(args.result)
    if root.is_dir():
        plot_archive(root, args.data, args.output, solution=args.solution)
    else:
        plot_file(root, args.data, args.output)




def draw_reserve_drt(solution, directory, dpi=200):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter
    warehouse_sources = {e.source for e in solution.excursions if e.split_trip == 'depot' and e.events}
    vehicles = sorted({d.vehicle for key, d in solution.duties.items() if key in warehouse_sources and any(v.kind == 'pickup' for v in d.visits)})
    if not vehicles:
        return None
    positions = {vehicle: i for i, vehicle in enumerate(vehicles)}
    paths, colors, rows = [], [], []
    occupied = {(ride.duty, i) for service in solution.services.values() for ride in service.rides for i in range(ride.board, ride.alight)}
    fig, ax = plt.subplots(figsize=(13, max(3, 0.4 * len(vehicles) + 1.5)))
    palette = {'drt': '#B5279B', 'fixed': '#208B55', 'solo': '#1FA6A6', 'empty': '#888888', 'couple': '#E7A128'}
    for duty in solution.duties.values():
        if duty.vehicle not in positions:
            continue
        y = positions[duty.vehicle]
        for index, leg in enumerate(duty.legs):
            kind = 'empty' if leg.kind == 'drt' and (duty.id, index) not in occupied else leg.kind
            paths.append([(leg.departure, y), (leg.arrival, y)])
            colors.append(palette.get(kind, '#888888'))
            rows.append([duty.vehicle, kind, leg.start, leg.end, leg.departure, leg.arrival, leg.distance])
    for move in solution.metadata.get('rebalance_moves', ()):
        if move['vehicle'] in positions:
            y = positions[move['vehicle']]
            paths.append([(move['departure'], y), (move['arrival'], y)])
            colors.append(palette['empty'])
            rows.append([move['vehicle'], 'empty', move['start'], move['end'], move['departure'], move['arrival'], move['distance']])
    ax.add_collection(LineCollection(paths, colors=colors, linewidths=3))
    ax.autoscale()
    ax.set_yticks(range(len(vehicles)), [f'MU {v}' for v in vehicles])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda t, _: f'{int(round(t * 60)) // 3600:02}:{int(round(t * 60)) // 60 % 60:02}'))
    ax.set_xlabel('Time')
    ax.set_ylabel('Physical module')
    ax.set_title('Warehouse DRT module schedule')
    ax.grid(axis='x', alpha=0.25)
    ax.legend(handles=[Line2D([0], [0], color=c, lw=3, label=k) for k, c in palette.items()], loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
    fig.tight_layout()
    file = Path(directory) / 'reserve_drt_schedule.png'
    fig.savefig(file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    with file.with_suffix('.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['vehicle', 'kind', 'start_node', 'end_node', 'departure', 'arrival', 'distance_km'])
        writer.writerows(sorted(rows, key=lambda row: (row[0], row[4])))
    return str(file)

if __name__ == '__main__':
    main()
