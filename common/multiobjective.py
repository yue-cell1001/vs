from bisect import bisect_left
from math import isfinite
from .pareto import objective_key

def layers2d(values):
    groups = {}
    for i, v in enumerate(values):
        if not all((isfinite(x) for x in v)):
            continue
        groups.setdefault(objective_key(v), []).append(i)
    ys = sorted({v[1] for v in groups})
    tree = [0] * (len(ys) + 1)
    layers = []
    for (x, y), indices in sorted(groups.items()):
        j = bisect_left(ys, y) + 1
        k = j
        rank = 0
        while k:
            rank = max(rank, tree[k])
            k -= k & -k
        if len(layers) <= rank:
            layers.append([])
        layers[rank].extend(indices)
        k = j
        while k < len(tree):
            tree[k] = max(tree[k], rank + 1)
            k += k & -k
    return layers

def hypervolume(values, upper, margin):
    if len(upper) != 2 or any((u <= 0 for u in upper)) or margin <= 0:
        raise ValueError('Invalid fixed HV scale')
    points = sorted({(v[0] / upper[0], v[1] / upper[1]) for v in (objective_key(x) for x in values)})
    r = 1 + margin
    last_y = r
    area = 0
    for x, y in points:
        if x < 0 or y < 0 or x > 1 + 1e-07 or (y > 1 + 1e-07):
            raise ValueError('Objective exceeds declared conservative bounds')
        if y < last_y:
            area += (r - x) * (last_y - y)
            last_y = y
    return area

def insertion_cost(delta, upper):
    return max((max(0, x) / u for x, u in zip(delta, upper)))
