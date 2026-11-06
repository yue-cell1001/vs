from collections import Counter, defaultdict
import numpy as np
from .routing import Infeasible


class CapacityGuard:

    def __init__(self, net, p):
        self.net, self.p = net, p
        self.positions = {line: {node: i for i, node in enumerate(stops)} for line, stops in net.lines.items()}
        releases = defaultdict(list)
        for person in net.passengers:
            if person.kind:
                continue
            a, b = self.positions[person.line][person.origin], self.positions[person.line][person.dest]
            for i in range(a, b):
                releases[person.line, i].append(person.earliest)
        self.demand = {key: np.asarray(sorted(values)) for key, values in releases.items()}
        self.calls = 0

    def check(self, trips, duties, excursions):
        self.calls += 1
        if len({e.source for e in excursions}) == len(excursions) and all(e.split_trip in trips and e.target in trips for e in excursions):
            from .initial_residual import scheduled_capacity
            scheduled_capacity(self, trips, excursions)
            return
        p, net = self.p, self.net
        forced = {e.source for e in excursions}
        counts = Counter((leg.trip, leg.start, leg.end, round(leg.departure, 7)) for key, duty in duties.items() if key in forced for leg in duty.legs if leg.kind == 'fixed')
        removed = Counter(source.rsplit('/m', 1)[0] for source in forced if not source.startswith('reserve:'))
        supply = defaultdict(list)
        for trip in trips.values():
            free = min([p.max_formation - removed[trip.id]] + [p.max_formation - counts[trip.id, leg.start, leg.end, round(leg.departure, 7)] for leg in trip.legs])
            if free < 0:
                raise Infeasible(f'{trip.id}: forced formation exceeds maximum')
            for i, leg in enumerate(trip.legs):
                amount = free + counts[trip.id, leg.start, leg.end, round(leg.departure, 7)]
                if amount < 1:
                    raise Infeasible(f'{trip.id}: no module left on a fixed segment')
                supply[trip.line, i].append((leg.departure, amount * p.capacity))
        for key in forced:
            duty = duties[key]
            for i, leg in enumerate(duty.legs):
                if leg.kind == 'solo':
                    position = self.positions[leg.line][leg.start]
                    supply[leg.line, position].append((leg.departure, p.capacity))
            anchor = None
            for i, visit in enumerate(duty.visits):
                if visit.node not in self.positions.get(visit.line, {}):
                    continue
                if anchor is not None:
                    a, first = anchor
                    begin = self.positions[first.line][first.node]
                    end = self.positions[visit.line][visit.node] if first.line == visit.line else -1
                    part = duty.legs[a:i]
                    if end > begin and any(leg.kind == 'drt' for leg in part) and not any(leg.kind == 'empty' for leg in part) and not any(v.kind in ('depot', 'stored_join') for v in duty.visits[a + 1:i]):
                        for j in range(begin, end):
                            supply[visit.line, j].append((visit.departure, p.capacity))
                anchor = None if visit.kind in ('depot', 'stored_join') else (i, visit)
        for key, arrivals in self.demand.items():
            opportunities = sorted(supply.get(key, ()))
            if not opportunities:
                raise Infeasible(f'Ordinary capacity protection: no supply on {key}')
            times = np.asarray([row[0] for row in opportunities])
            amounts = np.asarray([row[1] for row in opportunities])
            available = np.r_[np.cumsum(amounts[::-1])[::-1], 0]
            thresholds, first = np.unique(arrivals, return_index=True)
            required = len(arrivals) - first
            indices = np.searchsorted(times, thresholds - 1e-07)
            bad = np.flatnonzero(required > available[indices])
            if len(bad):
                j = int(bad[0])
                raise Infeasible(f'Ordinary capacity protection: {key[0]} segment {key[1] + 1}, time {thresholds[j]:.3f}, need {required[j]}, optimistic supply {available[indices[j]]}')
