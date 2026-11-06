from math import floor, ceil

def count_bounds(p):
    span = p.end - p.start
    return (ceil(span / p.max_headway), floor(span / p.min_headway))

def repair(times, p):
    import numpy as np
    from scipy.optimize import linprog
    low, high = count_bounds(p)
    n = max(low + 1, min(high + 1, len(times)))
    raw = sorted((max(p.start, min(p.end, float(t))) for t in times))
    if not raw:
        raw = [p.start, p.end]
    if len(raw) != n:
        raw = list(np.interp(np.linspace(0, len(raw) - 1, n), np.arange(len(raw)), raw))
    if raw[0] == p.start and raw[-1] == p.end and all((p.min_headway - 1e-09 <= b - a <= p.max_headway + 1e-09 for a, b in zip(raw, raw[1:]))):
        return raw
    rows = []
    rhs = []
    for i in range(n):
        row = np.zeros(2 * n)
        row[i] = 1
        row[n + i] = -1
        rows.append(row)
        rhs.append(raw[i])
        row = np.zeros(2 * n)
        row[i] = -1
        row[n + i] = -1
        rows.append(row)
        rhs.append(-raw[i])
    for i in range(n - 1):
        row = np.zeros(2 * n)
        row[i + 1] = 1
        row[i] = -1
        rows.append(row)
        rhs.append(p.max_headway)
        rows.append(-row)
        rhs.append(-p.min_headway)
    bounds = [(p.start, p.end)] * n + [(0, None)] * n
    bounds[0] = (p.start, p.start)
    bounds[n - 1] = (p.end, p.end)
    result = linprog([0] * n + [1] * n, A_ub=rows, b_ub=rhs, bounds=bounds, method='highs')
    if not result.success:
        raise ValueError('No legal timetable projection at selected length')
    return result.x[:n].tolist()

def violations(timetable, net, p):
    errors = []
    if set(timetable) != set(net.lines):
        errors.append('Timetable line set mismatch')
    for line, times in timetable.items():
        if not times or times[0] < p.start - 1e-06 or times[-1] > p.end + 1e-06:
            errors.append(f'{line}: outside planning horizon')
        if not times or abs(times[0] - p.start) > 1e-06 or abs(times[-1] - p.end) > 1e-06:
            errors.append(f'{line}: first/last service must cover the fixed horizon')
        low, high = count_bounds(p)
        if not low + 1 <= len(times) <= high + 1:
            errors.append(f'{line}: invalid trip count')
        if any((not p.min_headway - 1e-06 <= b - a <= p.max_headway + 1e-06 for a, b in zip(times, times[1:]))):
            errors.append(f'{line}: invalid headway')
    return errors
