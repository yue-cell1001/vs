from dataclasses import dataclass
from collections import defaultdict
from common.model import Ride
from .routing import Infeasible

@dataclass(frozen=True)
class Column:
    people: tuple[int, ...]
    rides: tuple[Ride, ...]
    pickup: float
    dropoff: float
    detour: float
    cost: float
    offroute: float = 0.0

def coupled(a, i, b, j):
    va, vb = (a.visits[i], b.visits[j])
    if not va.trip or (va.trip, va.node, va.departure) != (vb.trip, vb.node, vb.departure):
        return False
    if va.kind == 'stored_join':
        return False
    for offset in (-1, 0):
        x, y = (i + offset, j + offset)
        if 0 <= x < len(a.legs) and 0 <= y < len(b.legs):
            u, v = (a.legs[x], b.legs[y])
            if u.kind == v.kind == 'fixed' and (u.trip, u.start, u.end, u.departure) == (v.trip, v.start, v.end, v.departure):
                return True
    return False

def direct_duration(net, line, origin, dest, p):
    stops = net.lines[line]
    a, b = (stops.index(origin), stops.index(dest))
    return sum((net.road(x, y)[1] for x, y in zip(stops[a:b], stops[a + 1:b + 1]))) + (b - a - 1) * p.stop_time

def catalogue(net, groups, required, p, trips=None, excursions=()):
    from .routing import cross_line_spans
    crossings = cross_line_spans(groups, trips or {}, excursions)
    ordinary = defaultdict(list)
    pickups = defaultdict(list)
    drops = defaultdict(list)
    stations = defaultdict(list)
    for key, d in groups.items():
        for i, v in enumerate(d.visits):
            if v.line and v.trip:
                ordinary[v.line, v.node].append((key, i))
            if v.kind == 'pickup':
                pickups[v.node].append((key, i))
            if v.kind == 'dropoff':
                drops[v.node].append((key, i))
            if v.trip:
                stations[v.trip, v.node, v.departure].append((key, i))
    cohorts = defaultdict(list)
    for person in net.passengers:
        if not person.kind:
            cohorts[person.line, person.origin, person.dest, person.earliest].append(person)
    columns = []
    requests = []
    destinations = {}
    scheduled = set()
    for key, duty in groups.items():
        positions = defaultdict(list)
        for j, visit in enumerate(duty.visits):
            positions[visit.line, visit.node].append(j)
        destinations[key] = positions
        if (len({v.line for v in duty.visits}) == 1
                and all(v.kind == 'fixed' for v in duty.visits)
                and all(leg.kind == 'fixed' for leg in duty.legs)
                and not crossings.get(key)):
            scheduled.add(key)


    itineraries = {}
    for (line, origin, dest, earliest), people in cohorts.items():
        ids = tuple((x.id for x in people))
        opts = []
        route_key = (line, origin, dest)
        if route_key not in itineraries:
            itineraries[route_key] = ordinary_itineraries(
                net, groups, ordinary[line, origin], crossings, line, origin, dest, p,
                destinations, scheduled)
        for key, i, j, departure, arrival, detour, offroute in itineraries[route_key]:
            if departure < earliest - 1e-07:
                continue
            cost = departure - earliest + detour
            opts.append(Column(ids, (Ride(key, i, j),), departure, arrival, detour, cost, offroute))
        if not opts:
            raise Infeasible(f'No compatible ordinary itinerary for cohort {ids}')
        requests.append((len(people), tuple(range(len(columns), len(columns) + len(opts)))))
        columns.extend(opts)
    people = {x.id: x for x in net.passengers}
    for pid in sorted(required):
        person = people[pid]
        if len(pickups[person.origin]) != 1 or len(drops[person.dest]) != 1:
            raise Infeasible(f'DRT {pid}: pickup/delivery must each occur exactly once')
        key, i = pickups[person.origin][0]
        other, j = drops[person.dest][0]
        first, last = (groups[key], groups[other])
        pickup = first.visits[i].arrival
        dropoff = last.visits[j].arrival
        if not person.earliest - 1e-07 <= pickup <= person.latest + 1e-07 or dropoff < pickup - 1e-07:
            raise Infeasible(f'DRT {pid}: time window or precedence')
        opts = []
        cost = pickup - person.earliest
        if key == other and i < j and (not any((v.kind in ('stored_join', 'depot') for v in first.visits[i + 1:j]))):
            opts.append(Column((pid,), (Ride(key, i, j),), pickup, dropoff, 0, cost))
        if key != other:
            for a in range(i + 1, len(first.visits)):
                va = first.visits[a]
                if any((v.kind in ('stored_join', 'depot') for v in first.visits[i + 1:a])):
                    break
                for candidate, b in stations[va.trip, va.node, va.departure]:
                    if candidate != other or not b < j:
                        continue
                    if any((v.kind in ('stored_join', 'depot') for v in last.visits[b + 1:j])):
                        continue
                    if coupled(first, a, last, b):
                        opts.append(Column((pid,), (Ride(key, i, a), Ride(other, b, j)), pickup, dropoff, 0, cost))
        if not opts:
            raise Infeasible(f'DRT {pid}: no physically coupled <=1-transfer itinerary')
        requests.append((1, tuple(range(len(columns), len(columns) + len(opts)))))
        columns.extend(opts)
    return (columns, requests)


def ordinary_itineraries(net, groups, boarding, crossings, line, origin, dest, p,
                         destinations, scheduled):
    result = []
    baseline = direct_duration(net, line, origin, dest, p)
    for key, i in boarding:
        duty = groups[key]
        board = duty.visits[i]
        for j in destinations[key].get((line, dest), ()):
            if j <= i:
                continue
            end = duty.visits[j]
            if key not in scheduled:
                if any((v.line != line for v in duty.visits[i + 1:j + 1])):
                    continue
                if any((v.kind in ('depot', 'stored_join') for v in duty.visits[i + 1:j])):
                    continue
                if any((leg.kind == 'empty' for leg in duty.legs[i:j])):
                    continue
                if any((i < end and j > start for start, end in crossings.get(key, []))):
                    continue
            canonical = baseline
            detour = max(0, end.arrival - board.departure - canonical)
            if detour > p.max_detour + 1e-07:
                continue
            offroute = 0 if key in scheduled else sum((net.distance(leg.start, leg.end) / p.speed * 60 if leg.kind == 'drt' else net.road(leg.start, leg.end)[1] for leg in duty.legs[i:j] if leg.kind not in ('fixed', 'solo') and leg.start != leg.end))
            result.append((key, i, j, board.departure, end.arrival, detour, offroute))
    return result
