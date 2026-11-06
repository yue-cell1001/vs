from copy import deepcopy

def shifted_candidates(net, base, table, p, rng, released=None):
    positions = [(line, i) for line, times in table.items() for i in range(1, len(times) - 1) if released is None or (line, i) in released]
    rng.shuffle(positions)
    yield table
    for line, i in positions[:p.repair_budget]:
        times = table[line]
        low = max(base[line][i] - p.shift_limit, times[i - 1] + p.min_headway, times[i + 1] - p.max_headway)
        high = min(base[line][i] + p.shift_limit, times[i - 1] + p.max_headway, times[i + 1] - p.min_headway)
        if low > high + 1e-07:
            continue
        for value in sorted({low, high, (low + high) / 2}):
            if abs(value - times[i]) < 1e-07:
                continue
            trial = deepcopy(table)
            trial[line][i] = value
            yield trial

def validate_branch(table, base, p):
    if set(table) != set(base):
        return False
    for line, times in table.items():
        if len(times) != len(base[line]):
            return False
        if abs(times[0] - base[line][0]) > 1e-07 or abs(times[-1] - base[line][-1]) > 1e-07:
            return False
        if any((abs(a - b) > p.shift_limit + 1e-07 for a, b in zip(times, base[line]))):
            return False
        if any((not p.min_headway - 1e-07 <= b - a <= p.max_headway + 1e-07 for a, b in zip(times, times[1:]))):
            return False
    return True
