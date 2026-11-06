import numpy as np
from .constraints import repair

def profiles(net, p):
    edges = np.arange(p.start, p.end + p.demand_bin, p.demand_bin)
    result = {}
    for line in net.lines:
        demand = np.histogram([x.earliest for x in net.passengers if not x.kind and x.line == line], bins=edges)[0]
        low, high = (demand.min(), demand.max())
        normalized = (demand - low) / (high - low) if high > low else np.full(len(demand), 0.5)
        result[line] = p.max_headway - (p.max_headway - p.min_headway) * normalized
    return result

def guided(net, p, rng, reference):
    result = {}
    for line in net.lines:
        times = [p.start]
        while times[-1] < p.end - p.min_headway:
            index = min(len(reference[line]) - 1, int((times[-1] - p.start) / p.demand_bin))
            h = reference[line][index] + rng.uniform(-p.mutation_step, p.mutation_step)
            following = times[-1] + max(p.min_headway, min(p.max_headway, h))
            if following >= p.end:
                break
            times.append(following)
        if times[-1] != p.end:
            times.append(p.end)
        result[line] = repair(times, p)
    return result
