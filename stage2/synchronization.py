import numpy as np
from scipy.optimize import linprog
from .routing import Infeasible
from .service_limits import direct_drt_time


def service_times(net, people, events, origin, start, return_node, couple, p, completion=None):
    if not events:
        return ()
    cache = getattr(net, '_service_time_cache', None)
    if cache is None:
        cache = net._service_time_cache = {}
    windows = tuple((pid, pickup, people[pid].earliest, people[pid].latest) for pid, pickup in events)
    key = (windows, origin, start, return_node, couple, completion, p.speed, p.stop_time, p.max_coupling_wait, p.max_drt_detour)
    if key in cache:
        return cache[key]
    nodes = [people[pid].origin if pickup else people[pid].dest for pid, pickup in events]
    travel = [net.distance(origin, nodes[0]) / p.speed * 60] + [net.distance(a, b) / p.speed * 60 for a, b in zip(nodes, nodes[1:])]
    tail = net.distance(nodes[-1], return_node) / p.speed * 60 if return_node is not None else 0
    earliest = []
    time = start
    picked = {}
    pairs = []
    for i, (pid, pickup) in enumerate(events):
        time += travel[i]
        if pickup:
            time = max(time, people[pid].earliest)
            if time > people[pid].latest + 1e-7:
                raise Infeasible('Pickup window')
            picked[pid] = i
        elif pid in picked:
            pairs.append((picked[pid], i, direct_drt_time(net, people[pid], p) + p.max_drt_detour))
        earliest.append(time)
        time += p.stop_time
    for a, b, maximum in pairs:
        minimum = sum(travel[a + 1:b + 1]) + (b - a) * p.stop_time
        if minimum > maximum + 1e-7:
            raise Infeasible('DRT route travel exceeds direct time plus extra ride limit')
    arrival = time + tail
    coupled = couple is None or (arrival <= couple + 1e-7 and couple - arrival <= p.max_coupling_wait + 1e-7)
    bounded = completion is None or arrival <= completion + 1e-7
    rides = all(earliest[b] - earliest[a] <= maximum + 1e-7 for a, b, maximum in pairs)
    if coupled and bounded and rides:
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
        for a, b, maximum in pairs:
            row = np.zeros(n)
            row[b], row[a] = 1, -1
            rows.append(row)
            rhs.append(maximum)
        if couple is not None:
            row = np.zeros(n)
            row[-1] = 1
            rows.extend((row, -row))
            rhs.extend((couple - p.stop_time - tail, -(couple - p.max_coupling_wait - p.stop_time - tail)))
        if completion is not None:
            row = np.zeros(n)
            row[-1] = 1
            rows.append(row)
            rhs.append(completion - p.stop_time - tail)
        result = linprog([int(pickup) for _, pickup in events], A_ub=rows or None, b_ub=rhs or None,
                         A_eq=equal or None, b_eq=values or None, bounds=bounds, method='highs')
        if not result.success:
            raise Infeasible('Pickup, extra ride, coupling or completion limits are incompatible')
        answer = tuple(result.x.tolist())
    cache[key] = answer
    if len(cache) > 4096:
        cache.pop(next(iter(cache)))
    return answer
