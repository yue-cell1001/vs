from collections import defaultdict, Counter
from dataclasses import dataclass, replace
from heapq import heappop, heappush
import numpy as np
from common.integer import solve_integer
from scipy.sparse import coo_array
from common.model import Duty, Ride, Service, Solution
from .routing import Infeasible
from common.roads import RoadPaths
from .resources import inventory
from .passengers import catalogue

class Rows:

    def __init__(self):
        self.row = []
        self.col = []
        self.value = []
        self.lower = []
        self.upper = []

    def add(self, entries, lower=-np.inf, upper=np.inf):
        r = len(self.lower)
        for c, v in entries:
            if v:
                self.row.append(r)
                self.col.append(c)
                self.value.append(v)
        self.lower.append(lower)
        self.upper.append(upper)

    def matrix(self, n):
        return coo_array((np.asarray(self.value, float), (np.asarray(self.row, np.int32), np.asarray(self.col, np.int32))), shape=(len(self.lower), n)).tocsc()

def grouped_duties(duties, forced):
    groups = {}
    members = {}
    signatures = {}
    for key, duty in duties.items():
        signature = (tuple(duty.visits), tuple(duty.legs))
        if key in forced:
            name = key
        else:
            name = signatures.setdefault(signature, key)
        if name not in groups:
            groups[name] = duty
            members[name] = []
        members[name].append(key)
    return (groups, members)

class JointOptimizer:

    def __init__(self, net, p):
        self.net, self.p = (net, p)
        self.roads = RoadPaths(net, p)
        self.solves = 0
        self.initial_hint = None
        self.initial_seed_used = False
        self.objective_scales = None
        self.tradeoff_weights = (.5, .5)

    def solve(self, duties, trips, excursions, required, preference=0):
        p, net = (self.p, self.net)
        forced = {e.source for e in excursions}
        groups, members = grouped_duties(duties, forced)
        columns, requests = catalogue(net, groups, required, p, trips, excursions)
        names = list(groups)
        gi = {k: i for i, k in enumerate(names)}
        n = len(names)
        lower = []
        upper = []
        c1 = []
        c2 = []

        def variable(lo, hi, f1=0, f2=0):
            index = len(lower)
            lower.append(lo)
            upper.append(hi)
            c1.append(f1)
            c2.append(f2)
            return index
        use = {}
        birth = {}
        finish = {}
        chosen = {}
        links = []
        backup = {}
        rows = Rows()
        terminals = {stops[0] for stops in net.lines.values()}
        for key in names:
            duty = groups[key]
            marginal = sum((leg.distance * p.operating_cost * (p.following_ratio if leg.kind == 'fixed' else 1) for leg in duty.legs))
            use[key] = variable(1 if key in forced else 0, len(members[key]), marginal)
            birth[key] = variable(0, len(members[key]) if duty.visits[0].node in terminals or (key.startswith('reserve:') and duty.visits[0].node in net.depots) else 0, p.purchase_cost)
            finish[key] = variable(0, len(members[key]))
        for node in net.depots:
            rows.add([(birth[key], 1) for key in names if key.startswith('reserve:') and groups[key].visits[0].node == node], upper=p.depot_initial)
        for k, col in enumerate(columns):
            chosen[k] = variable(0, len(col.people), 0, col.cost)
        incoming = defaultdict(list)
        outgoing = defaultdict(list)
        for a, first in groups.items():
            end = first.visits[-1]
            for b, second in groups.items():
                start = second.visits[0]
                source_family = net.line_families.get(end.line, end.line)
                target_family = net.line_families.get(start.line, start.line)
                if source_family != target_family and not (end.node in terminals and start.node in terminals) and end.kind != 'depot' and start.kind != 'depot' and (end.node not in net.depots or start.node not in net.depots):
                    continue
                if not second.legs:
                    continue
                if abs(end.departure - start.departure) <= 1e-07 and (first.visits[0].departure, bool(first.legs), a) >= (start.departure, bool(second.legs), b):
                    continue
                if a == b or end.departure + p.operation_time > start.departure + 1e-07:
                    continue
                path = self.roads.reposition(end.node, start.node, start.departure - end.departure - p.operation_time)
                if path is None or end.departure + path[0] + p.operation_time > start.departure + 1e-07:
                    continue
                index = variable(0, min(len(members[a]), len(members[b])), path[1] * p.operating_cost)
                links.append((a, b, index, path))
                incoming[b].append(index)
                outgoing[a].append(index)
        for key in names:
            rows.add([(birth[key], 1), (use[key], -1)] + [(i, 1) for i in incoming[key]], 0, 0)
            rows.add([(finish[key], 1), (use[key], -1)] + [(i, 1) for i in outgoing[key]], 0, 0)
        for count, options in requests:
            rows.add([(chosen[k], 1) for k in options], count, count)
        load = defaultdict(list)
        for k, col in enumerate(columns):
            for ride in col.rides:
                for i in range(ride.board, ride.alight):
                    load[ride.duty, i].append((chosen[k], 1))
        for key, duty in groups.items():
            for i in range(len(duty.legs)):
                rows.add(load[key, i] + [(use[key], -p.capacity)], upper=0)
        formations = defaultdict(list)
        distances = {}
        planned = {(leg.trip, leg.start, leg.end, round(leg.departure, 7)) for t in trips.values() for leg in t.legs}
        for key, duty in groups.items():
            for leg in duty.legs:
                if leg.kind != 'fixed':
                    continue
                arc = (leg.trip, leg.start, leg.end, round(leg.departure, 7))
                formations[arc].append((use[key], 1))
                distances[arc] = leg.distance
        constant = 0.0
        for arc in planned:
            if arc not in formations:
                raise Infeasible('Fixed trip has no module templates')
        for arc, entries in formations.items():
            if arc in planned:
                rows.add(entries, 1, p.max_formation)
                constant += distances[arc] * p.operating_cost * (1 - p.following_ratio)
            else:
                z = variable(0, 1, distances[arc] * p.operating_cost * (1 - p.following_ratio))
                backup[z] = entries
                rows.add(entries + [(z, -p.max_formation)], upper=0)
                rows.add(entries + [(z, -1)], lower=0)
        warehouse_spans = defaultdict(list)
        for key, duty in groups.items():
            for v in duty.visits:
                if v.kind == 'stored_join':
                    a, b = (v.arrival + p.operation_time, v.departure - p.operation_time)
                    if b < a - 1e-07:
                        raise Infeasible('Negative storage interval')
                    if b > a + 1e-07:
                        warehouse_spans[v.node].append((a, b, use[key]))
            for v in duty.visits[:-1]:
                if v.kind == 'depot' and v is not duty.visits[0] and v.departure > v.arrival + p.operation_time + 1e-07:
                    warehouse_spans[v.node].append((v.arrival + p.operation_time, v.departure, use[key]))
            end = duty.visits[-1]
            if end.kind == 'depot':
                warehouse_spans[end.node].append((end.departure, np.inf, finish[key]))
        for a, b, index, path in links:
            end, start = groups[a].visits[-1], groups[b].visits[0]
            ready = end.departure + path[0]
            if start.node in net.depots and ready < start.departure - 1e-07:
                warehouse_spans[start.node].append((ready, start.departure, index))
        for node in net.depots:
            spans = warehouse_spans[node]
            departures = [(groups[key].visits[0].departure, birth[key]) for key in names if key.startswith('reserve:') and groups[key].visits[0].node == node]
            times = {p.start} | {a for a, _, _ in spans} | {b for _, b, _ in spans if np.isfinite(b)} | {time for time, _ in departures}
            for time in sorted(times):
                entries = [(idx, 1) for a, b, idx in spans if a <= time < b]
                entries += [(idx, -1) for release, idx in departures if release <= time]
                rows.add(entries, -p.depot_initial, p.depot_capacity - p.depot_initial)
        costs1 = np.asarray(c1)
        costs2 = np.asarray(c2)
        matrix = rows.matrix(len(lower))
        seed = self.incumbent(groups, members, columns, requests, use, birth, finish, chosen, links, lower, upper, matrix, rows.lower, rows.upper, forced, backup)
        from common.integer import _verified
        verified_seed = _verified(seed, lower, upper, matrix, rows.lower, rows.upper) if getattr(self, 'verify_only', False) else None
        if verified_seed is not None:
            x = np.rint(verified_seed).astype(int)
            return self.materialize(groups, members, columns, chosen, use, birth, finish, links, x, excursions, {'status': 'feasible_verified_not_optimized', 'primary_gap': None, 'F1_linear': float(costs1 @ x + constant), 'F2_linear': float(costs2 @ x), 'initial_reservation_seed': self.initial_seed_used})
        from .scalarization import build_problem
        if self.objective_scales is None:
            from .bounds import objective_bounds
            table = {}
            for trip in trips.values():
                table.setdefault(trip.line, []).append(trip.visits[0].arrival)
            table = {line: sorted(set(times)) for line, times in table.items()}
            self.objective_scales = objective_bounds(net, [{'timetable': table}], p)[0]
        problem = build_problem(costs1, costs2, constant, lower, upper, matrix,
                                rows.lower, rows.upper, seed, self.objective_scales, self.tradeoff_weights)
        cost, lo, hi, full, lb, ub, warm, integrality, weights = problem
        from .budget import check
        from time import monotonic
        check(self)
        seconds = p.subproblem_seconds
        if getattr(self, 'deadline', None) is not None:
            remaining = self.deadline - monotonic()
            seconds = min(seconds, remaining) if seconds is not None else remaining
        result = solve_integer(cost, lo, hi, full, lb, ub, seconds, warm, integrality=integrality)
        self.solves += 1
        if result.x is None:
            raise Infeasible(f'Joint integer allocation: {result.message}; no verified incumbent')
        x = np.rint(result.x[:-1]).astype(int)
        return self.materialize(groups, members, columns, chosen, use, birth, finish, links, x, excursions, {
            'status': 'scalar_optimal' if result.optimal else 'feasible_not_proven_optimal',
            'optimization_method': 'augmented_weighted_chebyshev',
            'optimality_scope': 'fixed_routes_and_recorded_direction',
            'objective_scales': list(self.objective_scales), 'tradeoff_weights': list(weights),
            'augmentation': .001, 'scalar_gap': result.gap, 'scalar_status': result.message,
            'scalar_lower_bound': result.lower_bound, 'scalar_value': float(cost @ result.x),
            'variables': len(lo), 'constraints': full.shape[0], 'solver_calls': self.solves,
            'solve_calls_this_allocation': 1, 'initial_reservation_seed': self.initial_seed_used,
            'F1_linear': float(costs1 @ x + constant), 'F2_linear': float(costs2 @ x)})

    def incumbent(self, groups, members, columns, requests, use, birth, finish, chosen, links, lower, upper, matrix, lb, ub, forced, backup):
        p = self.p
        from .initial_seed import reservation_seed
        seed = reservation_seed(self.initial_hint, groups, members, columns, requests, use, birth, finish, chosen, lower, upper, matrix, lb, ub)
        self.initial_seed_used = seed is not None
        if seed is not None:
            return seed
        x = np.zeros(len(lower))
        planned = defaultdict(list)
        for key, duty in groups.items():
            for leg in duty.legs:
                if leg.kind == 'fixed':
                    planned[leg.trip, leg.start, leg.end, leg.departure].append(key)
        forced_count = {arc: sum((key in forced for key in keys)) for arc, keys in planned.items()}
        for key, duty in groups.items():
            count = 0 if not duty.legs and key not in forced else 1 if key in forced else min([len(members[key])] + [p.max_formation - forced_count[leg.trip, leg.start, leg.end, leg.departure] for leg in duty.legs if leg.kind == 'fixed'])
            if count < 0:
                return None
            x[use[key]] = count
        available = {(key, i): int(p.capacity * x[use[key]]) for key, duty in groups.items() for i in range(len(duty.legs))}
        people = {person.id: person for person in self.net.passengers}
        ordered = sorted(requests, key=lambda req: (not people[columns[req[1][0]].people[0]].kind, people[columns[req[1][0]].people[0]].earliest))
        for count, opts in ordered:
            remaining = count
            for k in sorted(opts, key=lambda k: columns[k].cost):
                col = columns[k]
                spans = [(r.duty, i) for r in col.rides for i in range(r.board, r.alight)]
                take = min([remaining] + [available[span] for span in spans])
                if take:
                    x[chosen[k]] += take
                    for span in spans:
                        available[span] -= take
                    remaining -= take
                if not remaining:
                    break
            if remaining:
                return None
        candidates = defaultdict(list)
        for a, b, index, path in links:
            candidates[b].append((a, index, path))
        endings = {}
        for key in sorted(groups, key=lambda k: groups[k].visits[0].departure):
            quantity = int(x[use[key]])
            missing = quantity
            for a, index, path in sorted(candidates[key], key=lambda v: (groups[v[0]].visits[-1].kind != 'depot', v[2][1])):
                take = min(missing, endings.get(a, 0))
                if take:
                    x[index] += take
                    endings[a] -= take
                    missing -= take
                if not missing:
                    break
            x[birth[key]] = missing
            endings[key] = quantity
        for key, count in endings.items():
            x[finish[key]] = count
        for indicator, entries in backup.items():
            x[indicator] = int(any((x[index] > 0 for index, _ in entries)))
        residual = matrix @ x
        if np.any(residual < np.asarray(lb) - 1e-06) or np.any(residual > np.asarray(ub) + 1e-06):
            return None
        return x

    def materialize(self, groups, members, columns, chosen, use, birth, finish, links, x, excursions, solver):
        p, net = (self.p, self.net)
        selected = {}
        concrete = {}
        for key, duty in groups.items():
            ids = members[key][:x[use[key]]]
            concrete[key] = ids
            for name in ids:
                selected[name] = Duty(name, list(duty.visits), list(duty.legs))
        services = {}
        intervals = defaultdict(list)
        people = {q.id: q for q in net.passengers}
        next_person = defaultdict(int)
        for k, col in enumerate(columns):
            count = x[chosen[k]]
            if not count:
                continue
            cohort = col.people
            begin = next_person[cohort]
            assigned = cohort[begin:begin + count]
            next_person[cohort] += count
            for pid in assigned:
                if people[pid].kind:
                    services[pid] = Service(pid, col.rides, col.pickup, col.dropoff, col.detour, col.offroute)
                else:
                    intervals[col.rides[0].duty].append((col.rides[0].board, col.rides[0].alight, pid, col))
        for key, rides in intervals.items():
            free = list(range(p.capacity * len(concrete[key])))
            occupied = []
            for a, b, pid, col in sorted(rides, key=lambda v: (v[0], v[1], v[2])):
                while occupied and occupied[0][0] <= a:
                    _, seat = heappop(occupied)
                    heappush(free, seat)
                if not free:
                    raise Infeasible('Integral interval colouring failed')
                seat = heappop(free)
                heappush(occupied, (b, seat))
                duty = concrete[key][seat // p.capacity]
                services[pid] = Service(pid, (Ride(duty, a, b),), col.pickup, col.dropoff, col.detour, col.offroute)
        successors = defaultdict(list)
        predecessors = defaultdict(list)
        for a, b, index, path in links:
            if x[index]:
                predecessors[b].append((a, x[index], path))
        pool = defaultdict(list)
        chains = defaultdict(list)
        activations = []
        moves = []
        ends = {}
        next_id = 0
        storage = []
        for key in sorted(groups, key=lambda k: (groups[k].visits[0].departure, bool(groups[k].legs), k)):
            assignments = []
            arrivals = {}
            for a, count, path in predecessors[key]:
                for _ in range(count):
                    vehicle = pool[a].pop()
                    t = ends[vehicle][1]
                    for u, v, distance, travel in path[2]:
                        moves.append({'vehicle': vehicle, 'start': u, 'end': v, 'departure': t, 'arrival': t + travel, 'distance': distance, 'reason': 'optimal_net_rebalance'})
                        t += travel
                    arrivals[vehicle] = t
                    start = groups[key].visits[0]
                    if start.node in net.depots and t < start.departure - 1e-07:
                        storage.append({'node': start.node, 'vehicle': vehicle, 'in': t, 'out': start.departure, 'kind': 'between_duties'})
                    assignments.append(vehicle)
            for _ in range(x[birth[key]]):
                vehicle = next_id
                next_id += 1
                assignments.append(vehicle)
                v = groups[key].visits[0]
                activations.append({'vehicle': vehicle, 'node': v.node, 'time': v.departure, 'origin': 'reserve' if key.startswith('reserve:') else 'terminal'})
            if len(assignments) != len(concrete[key]):
                raise Infeasible('Module flow materialization mismatch')
            for name, vehicle in zip(concrete[key], assignments):
                selected[name].vehicle = vehicle
                first = selected[name].visits[0]
                if vehicle in arrivals and first.kind == 'depot':
                    selected[name].visits[0] = replace(first, arrival=arrivals[vehicle])
                chains[vehicle].append(name)
                pool[key].append(vehicle)
                last = selected[name].visits[-1]
                ends[vehicle] = (last.node, last.departure, last.kind)
        for vehicle, (node, time, kind) in ends.items():
            if kind == 'depot':
                storage.append({'node': node, 'vehicle': vehicle, 'in': time, 'out': None, 'kind': 'final'})
        for duty in selected.values():
            for v in duty.visits[:-1]:
                if v.kind == 'depot' and v is not duty.visits[0] and v.departure > v.arrival + p.operation_time + 1e-07:
                    storage.append({'node': v.node, 'vehicle': duty.vehicle, 'in': v.arrival + p.operation_time, 'out': v.departure, 'kind': 'warehouse_continuation'})
            for v in duty.visits:
                if v.kind == 'stored_join':
                    storage.append({'node': v.node, 'vehicle': duty.vehicle, 'in': v.arrival + p.operation_time, 'out': v.departure - p.operation_time, 'kind': 'join'})
        solution = Solution(selected, services, tuple(excursions))
        solution.metadata = {'fleet_size': next_id, 'vehicle_chains': dict(chains), 'activations': activations, 'rebalance_moves': moves, 'storage': storage, 'inventory': inventory(storage, net, p, activations), 'trim_offsets': {}, 'idle_tail_releases': [], 'joint_solver': solver}
        return solution
