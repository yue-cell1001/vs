from bisect import bisect_left
from collections import OrderedDict
from common.timetable import make_trips
from .constraints import repair, count_bounds


class CapacityAllocation:

    def __init__(self, net, p):
        self.net, self.p = net, p
        self.people = {line: [] for line in net.lines}
        for q in net.passengers:
            if not q.kind:
                stops = net.lines[q.line]
                self.people[q.line].append((stops.index(q.origin), q.earliest, q.id, stops.index(q.dest)))
        for people in self.people.values():
            people.sort()
        self.cache = {line: OrderedDict() for line in net.lines}

    def line(self, line, times):
        key = tuple(times)
        cache = self.cache[line]
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        trips = list(make_trips(self.net, {line: times}, self.p).values())
        loads = [[0] * len(t.legs) for t in trips]
        departures = [[t.visits[i].departure for t in trips] for i in range(len(self.net.lines[line]))]
        assignments, missing = {}, []
        for a, earliest, pid, b in self.people[line]:
            k = bisect_left(departures[a], earliest - 1e-7)
            while k < len(trips) and any(loads[k][i] >= self.p.capacity * self.p.max_formation for i in range(a, b)):
                k += 1
            if k == len(trips):
                missing.append(pid)
                continue
            t = trips[k]
            assignments[pid] = {'trip': t.id, 'board': t.visits[a].departure, 'alight': t.visits[b].arrival, 'waiting': max(0, t.visits[a].departure - earliest)}
            for i in range(a, b):
                loads[k][i] += 1
        result = assignments, {t.id: load for t, load in zip(trips, loads)}, missing
        cache[key] = result
        if len(cache) > 64:
            cache.popitem(last=False)
        return result

    def repair(self, table):
        result = {line: list(times) for line, times in table.items()}
        _, high = count_bounds(self.p)
        for line, times in result.items():
            current = self.line(line, times)
            for _ in range(self.p.repair_budget):
                if not current[2] or len(times) >= high + 1:
                    break
                trials = []
                for a, b in zip(times, times[1:]):
                    if b - a <= self.p.min_headway + 1e-7:
                        continue
                    raw = sorted(times + [(a + b) / 2])
                    candidate = repair(raw, self.p)
                    value = self.line(line, candidate)
                    rank = (len(value[2]), sum(q['waiting'] for q in value[0].values()))
                    trials.append((rank, candidate, value))
                if not trials:
                    break
                _, times, current = min(trials, key=lambda x: x[0])
                result[line] = times
        return result
