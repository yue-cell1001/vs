from collections import defaultdict
from .routing import Infeasible

def inventory(storage, net, p, activations=()):
    spans = defaultdict(list)
    for entry in storage:
        end = entry['out']
        if end is not None and end < entry['in'] - 1e-07:
            raise Infeasible('Negative storage interval')
        if end is None or end > entry['in'] + 1e-07:
            spans[entry['vehicle']].append((entry['in'], float('inf') if end is None else end))
    for intervals in spans.values():
        ordered = sorted(intervals)
        if any((a[1] > b[0] + 1e-07 for a, b in zip(ordered, ordered[1:]))):
            raise Infeasible('One module stored in overlapping warehouse intervals')
    events = defaultdict(list)
    for s in storage:
        if s['node'] not in net.depots:
            raise Infeasible('Storage outside warehouse')
        if s['out'] is not None and s['out'] <= s['in'] + 1e-07:
            continue
        events[s['node']].append((s['in'], 1, s['vehicle']))
        if s['out'] is not None:
            events[s['node']].append((s['out'], -1, s['vehicle']))
    activated = defaultdict(int)
    for item in activations:
        if item.get('origin') == 'reserve':
            if item['node'] not in net.depots or item['time'] < p.start - 1e-07:
                raise Infeasible('Invalid reserve activation')
            activated[item['node']] += 1
            events[item['node']].append((item['time'], -1, item['vehicle']))
    if any(count > p.depot_initial for count in activated.values()):
        raise Infeasible('Reserve activations exceed initial stock')
    report = {}
    for node in net.depots:
        count = peak = p.depot_initial
        history = []
        for time, change, vehicle in sorted(events[node], key=lambda e: (e[0], e[1])):
            count += change
            peak = max(peak, count)
            if not 0 <= count <= p.depot_capacity:
                raise Infeasible(f'Warehouse {node} inventory {count}')
            history.append({'time': time, 'change': change, 'inventory': count, 'vehicle': vehicle})
        report[node] = {'initial': p.depot_initial, 'final': count, 'peak': peak, 'events': history}
    return report
