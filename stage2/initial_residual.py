from collections import Counter
from .routing import Infeasible


def scheduled_capacity(decoder, trips, plan):
    import numpy as np
    net, p = decoder.net, decoder.p
    demand_cache = getattr(decoder, 'demand_cache', {})
    leaves = Counter((e.split_trip, e.split) for e in plan)
    joins = Counter((e.target, e.join) for e in plan)
    forced = Counter(e.split_trip for e in plan)
    upper = {}
    for key, trip in trips.items():
        lo = max(1, forced[key])
        hi = p.max_formation
        change = 0
        offsets = []
        for i in range(len(trip.visits)):
            lo = max(lo, leaves[key, i] - change)
            hi = min(hi, p.max_formation - change)
            change += joins[key, i]
            hi = min(hi, p.max_formation - change)
            change -= leaves[key, i]
            lo = max(lo, (1 if i < len(trip.visits) - 1 else 0) - change)
            hi = min(hi, p.max_formation - change)
            offsets.append(change)
        if lo > hi:
            raise Infeasible('Split/join formation capacity')
        upper[key] = [hi + v for v in offsets[:-1]]
    for line, stops in net.lines.items():
        duties = sorted((t for t in trips.values() if t.line == line), key=lambda t: int(t.id.rsplit(':', 1)[1]))
        n = len(duties)
        signature = (line, tuple(tuple(v.departure for v in t.visits) for t in duties))
        shared = getattr(decoder, 'ordinary_demand_cache', None)
        if shared is None:
            shared = decoder.ordinary_demand_cache = {}
        demand = demand_cache.get(line)
        if demand is None:
            demand = shared.get(signature)
        if demand is None:
            cohorts = getattr(decoder, 'ordinary_cohorts', None)
            if cohorts is None:
                cohorts = decoder.ordinary_cohorts = {}
                for q in net.passengers:
                    if not q.kind:
                        route = net.lines[q.line]
                        cohorts.setdefault(q.line, []).append((route.index(q.origin), route.index(q.dest), q.earliest))
            demand = np.zeros((n, len(stops)), int)
            departures = np.asarray([[v.departure for v in t.visits] for t in duties])
            for origin, dest, arrival in cohorts.get(line, ()):
                eligible = np.flatnonzero(departures[:, origin] >= arrival - 1e-7)
                if not len(eligible):
                    raise Infeasible('No last-trip ordinary coverage')
                first = int(eligible[0])
                demand[first, origin] += 1
                demand[first, dest] -= 1
            demand = np.cumsum(np.cumsum(demand, axis=1)[:, :-1][::-1], axis=0)[::-1]
            if len(shared) >= 512:
                shared.pop(next(iter(shared)))
            shared[signature] = demand
        demand_cache[line] = demand
        cap = np.array([upper[t.id] for t in duties])
        for e in plan:
            if trips[e.split_trip].line == line and trips[e.target].line == line:
                cap[int(e.split_trip.rsplit(':', 1)[1]), e.split:e.join] += 1
        if np.any(demand > p.capacity * np.cumsum(cap[::-1], axis=0)[::-1]):
            raise Infeasible('Ordinary residual capacity protection')
    return trips
