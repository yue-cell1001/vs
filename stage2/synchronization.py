import numpy as np
from scipy.optimize import linprog
from .routing import Infeasible


def service_times(net, people, events, origin, start, return_node, couple, p):
    if not events:
        return ()
    cache = getattr(net, '_service_time_cache', None)
    if cache is None:
        cache = net._service_time_cache = {}
    windows = tuple((pid, pickup, people[pid].earliest, people[pid].latest) for pid, pickup in events)
    key = (windows, origin, start, return_node, couple, p.speed, p.stop_time, p.max_coupling_wait)
    if key in cache:
        return cache[key]
    nodes = [people[pid].origin if pickup else people[pid].dest for pid, pickup in events]
    travel = [net.distance(origin, nodes[0]) / p.speed * 60] + [net.distance(a, b) / p.speed * 60 for a, b in zip(nodes, nodes[1:])]
    tail = net.distance(nodes[-1], return_node) / p.speed * 60
    earliest = []
    time = start
    for i, (pid, pickup) in enumerate(events):
        time += travel[i]
        if pickup:
            time = max(time, people[pid].earliest)
            if time > people[pid].latest + 1e-07:
                raise Infeasible('Pickup window')
        earliest.append(time)
        time += p.stop_time
    arrival = time + tail
    if arrival <= couple + 1e-07 and couple - arrival <= p.max_coupling_wait + 1e-07:
        answer = tuple(earliest)
    else:
        n = len(events)
        rows, rhs, equal, values = [], [], [], []
        bounds = [(people[pid].earliest, people[pid].latest) if pickup else (start, None) for pid, pickup in events]
        for i, (_, pickup) in enumerate(events):
            row = np.zeros(n)
            row[i] = -1
            if i:
                row[i - 1] = 1
                value = -p.stop_time - travel[i]
            else:
                value = -start - travel[0]
            if pickup:
                rows.append(row)
                rhs.append(value)
            else:
                equal.append(row)
                values.append(value)
        row = np.zeros(n)
        row[-1] = 1
        rows.extend((row, -row))
        rhs.extend((couple - p.stop_time - tail, -(couple - p.max_coupling_wait - p.stop_time - tail)))
        result = linprog([int(pickup) for _, pickup in events], A_ub=rows, b_ub=rhs, A_eq=equal or None, b_eq=values or None, bounds=bounds, method='highs')
        if not result.success:
            raise Infeasible('Pickup windows cannot synchronize with return coupling')
        answer = tuple(result.x.tolist())
    cache[key] = answer
    if len(cache) > 4096:
        cache.pop(next(iter(cache)))
    return answer
