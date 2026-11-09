from collections import Counter, defaultdict
from .routing import Infeasible

def arc_key(leg):
    return (leg.trip, leg.start, leg.end, round(leg.departure, 7))

class DeltaEvaluator:

    def __init__(self, net, p):
        self.net, self.p = (net, p)
        self.parts = {}
        self.counts = Counter()
        self.distances = {}
        self.fixed_cost = 0
        self.variable = 0
        self.passenger_parts = {}
        self.people = {q.id: q for q in net.passengers}
        self.sums = [0.0] * 4

    def fixed(self, key, count):
        if not count:
            return 0
        p = self.p
        return self.distances[key] * p.operating_cost * (1 + p.following_ratio * (count - 1))

    def update(self, solution):
        p = self.p
        new = {}
        changed = set()
        touched = set()
        reused = 0
        for key, duty in solution.duties.items():
            footprint = tuple(duty.legs)
            if key in self.parts and self.parts[key][0] == footprint:
                new[key] = self.parts[key]
                reused += 1
                continue
            counts = Counter()
            cost = 0.0
            for leg in duty.legs:
                if leg.kind == 'fixed':
                    a = arc_key(leg)
                    counts[a] += 1
                    self.distances[a] = leg.distance
                else:
                    cost += leg.distance * p.operating_cost
            new[key] = (footprint, counts, cost)
            changed.add(key)
        removed = set(self.parts) - set(new)
        changed |= removed
        for key in changed:
            if key in self.parts:
                touched.update(self.parts[key][1])
            if key in new:
                touched.update(new[key][1])
        previous = sum((self.fixed(a, self.counts[a]) for a in touched))
        for key in changed:
            if key in self.parts:
                _, counts, cost = self.parts[key]
                self.counts.subtract(counts)
                self.variable -= cost
            if key in new:
                _, counts, cost = new[key]
                self.counts.update(counts)
                self.variable += cost
        self.fixed_cost += sum((self.fixed(a, self.counts[a]) for a in touched)) - previous
        self.parts = new
        changed_people = set(self.passenger_parts) - set(solution.services)
        reused_people = 0
        parts = {}
        for pid, service in solution.services.items():
            if pid in self.passenger_parts and self.passenger_parts[pid][0] == service:
                parts[pid] = self.passenger_parts[pid]
                reused_people += 1
                continue
            person = self.people[pid]
            wait = max(0, service.pickup - person.earliest)
            values = (0, wait, 0, 0) if person.kind else (wait, 0, service.detour, service.offroute_minutes)
            parts[pid] = (service, values)
            changed_people.add(pid)
        for pid in changed_people:
            if pid in self.passenger_parts:
                for i, value in enumerate(self.passenger_parts[pid][1]):
                    self.sums[i] -= value
            if pid in parts:
                for i, value in enumerate(parts[pid][1]):
                    self.sums[i] += value
        self.passenger_parts = parts
        normal, drt, quality, offroute = self.sums
        rebalance = sum((m['distance'] for m in solution.metadata['rebalance_moves'])) * p.operating_cost
        purchase = solution.metadata['fleet_size'] * p.purchase_cost
        solo = sum(leg.distance * p.operating_cost for duty in solution.duties.values() for leg in duty.legs if leg.kind == 'solo')
        solution.components = {'purchase': purchase, 'fixed_travel': self.fixed_cost, 'drt_travel': self.variable - solo, 'solo_travel': solo, 'rebalance_travel': rebalance, 'normal_wait_minutes': normal, 'drt_wait_minutes': drt, 'normal_detour_minutes': quality, 'normal_offroute_minutes': offroute, 'drt_detour_minutes': sum(s.detour for pid, s in solution.services.items() if self.people[pid].kind)}
        solution.objectives = (purchase + self.fixed_cost + self.variable + rebalance, normal + drt + quality)
        solution.metadata['incremental_evaluation'] = {'changed_duties': len(changed), 'reused_duties': reused, 'changed_passengers': len(changed_people), 'reused_passengers': reused_people, 'changed_fixed_arcs': len(touched)}
        return solution.objectives

class DeltaAudit:

    def __init__(self, net, p):
        self.net, self.p = (net, p)
        self.paths = {}
        self.services = {}

    def check(self, solution):
        paths = {k: (tuple(d.visits), tuple(d.legs)) for k, d in solution.duties.items()}
        dirty = {k for k, v in paths.items() if self.paths.get(k) != v} | set(self.paths) - set(paths)
        changed = {pid for pid, s in solution.services.items() if self.services.get(pid) != s}
        for pid, s in solution.services.items():
            if any((r.duty in dirty for r in s.rides)):
                changed.add(pid)
        load = {key: [0] * len(solution.duties[key].legs) for key in dirty if key in solution.duties}
        for pid, s in solution.services.items():
            for ride in s.rides:
                if ride.duty in load:
                    for i in range(ride.board, ride.alight):
                        load[ride.duty][i] += 1
        for key, loads in load.items():
            if any((n > self.p.capacity for n in loads)):
                raise Infeasible('Incremental capacity violation')
            duty = solution.duties[key]
            if len(duty.visits) != len(duty.legs) + 1:
                raise Infeasible('Incremental path cardinality violation')
            for i, leg in enumerate(duty.legs):
                a, b = duty.visits[i:i + 2]
                if (leg.start, leg.end) != (a.node, b.node) or abs(leg.departure - a.departure) > 1e-07 or abs(leg.arrival - b.arrival) > 1e-07:
                    raise Infeasible('Incremental path continuity violation')
                if leg.arrival < leg.departure - 1e-07 or a.departure < a.arrival - 1e-07:
                    raise Infeasible('Incremental time violation')
        people = {p.id: p for p in self.net.passengers}
        for pid in changed:
            person = people[pid]
            service = solution.services[pid]
            if service.pickup < person.earliest - 1e-07 or service.dropoff < service.pickup - 1e-07:
                raise Infeasible('Incremental passenger precedence violation')
            if person.kind and service.pickup > person.latest + 1e-07:
                raise Infeasible('Incremental pickup window violation')
            if len(service.rides) > (2 if person.kind else 1):
                raise Infeasible('Incremental transfer count violation')
            if not person.kind and service.detour > self.p.max_detour + 1e-07:
                raise Infeasible('Incremental maximum detour violation')
        self.paths = paths
        self.services = dict(solution.services)
        solution.metadata['incremental_checks'] = {'examined_duties': len(load), 'reused_duties': len(paths) - len(load), 'examined_passengers': len(changed), 'reused_passengers': len(solution.services) - len(changed)}
