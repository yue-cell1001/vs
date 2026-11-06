from dataclasses import dataclass
from math import ceil, inf
from common.timetable import make_trips
from common.roads import RoadPaths
from .capacity import CapacityAllocation

@dataclass
class Reference:
    objectives: tuple
    trips: dict
    assignments: dict
    modules: list
    reposition: list
    uncovered: list
    capacity_risks: list

    def summary(self):
        return {'objectives': self.objectives, 'trips': self.trips, 'assignments': self.assignments, 'modules': self.modules, 'reposition': self.reposition, 'uncovered': self.uncovered, 'capacity_risks': self.capacity_risks, 'reference_only_not_stage2_feasibility': True, 'ordinary_capacity_feasible': not self.uncovered}

class ReferenceDecoder:

    def __init__(self, net, p):
        self.net, self.p = (net, p)
        self.roads = RoadPaths(net, p)
        self.allocation = CapacityAllocation(net, p)

    def decode(self, timetable):
        net, p = (self.net, self.p)
        trips = make_trips(net, timetable, p)
        loads, assignments, uncovered = {}, {}, []
        for line in net.lines:
            assigned, line_loads, missing = self.allocation.line(line, timetable[line])
            loads.update(line_loads)
            assignments.update(assigned)
            uncovered.extend(missing)
        waiting = sum(q['waiting'] for q in assignments.values())
        module_pool = [{'id': index * p.depot_initial + slot, 'family': None, 'node': node, 'available': p.start - p.turnaround_time, 'tasks': [], 'origin': 'reserve', 'home_depot': node} for index, node in enumerate(net.depots) for slot in range(p.depot_initial)]
        records = {}
        moves = []
        risks = []
        distance_cost = 0
        for key, trip in sorted(trips.items(), key=lambda kv: (kv[1].visits[0].arrival, kv[0])):
            peak = max(loads[key], default=0)
            count = max(1, min(p.max_formation, ceil(peak / (p.reference_load * p.capacity))))
            first, last = (trip.visits[0], trip.visits[-1])
            family = net.line_families.get(trip.line, trip.line)
            candidates = []
            for module in module_pool:
                path = self.roads.reposition(module['node'], first.node, first.arrival - module['available'] - p.turnaround_time)
                if path is not None and module['available'] + path[0] + p.turnaround_time <= first.arrival + 1e-07:
                    candidates.append((0 if module['tasks'] else 1, path[1], module['id'], module, path))
            selected = []
            for _, _, _, module, path in sorted(candidates, key=lambda x: x[:3])[:count]:
                selected.append(module)
                moves.append({'vehicle': module['id'], 'from': module['node'], 'to': first.node, 'distance': path[1], 'departure': module['available'] + p.turnaround_time, 'arrival': module['available'] + p.turnaround_time + path[0]})
            while len(selected) < count:
                module = {'id': len(module_pool), 'family': family, 'node': first.node, 'available': first.arrival, 'tasks': []}
                module_pool.append(module)
                selected.append(module)
            for module in selected:
                module['family'] = family
                module['node'] = last.node
                module['available'] = last.departure
                module['tasks'].append(key)
            length = sum((l.distance for l in trip.legs))
            distance_cost += p.operating_cost * length * (1 + p.following_ratio * (count - 1))
            records[key] = {'line': trip.line, 'start': first.arrival, 'end': last.departure, 'peak': peak, 'loads': loads[key], 'formation': count, 'vehicles': [m['id'] for m in selected], 'distance': length}
        module_pool = [module for module in module_pool if module['tasks']]
        cost = p.purchase_cost * len(module_pool) + distance_cost + p.operating_cost * sum(move['distance'] for move in moves)
        return Reference((inf, inf) if uncovered else (cost, waiting), records, assignments, module_pool, moves, uncovered, risks)
