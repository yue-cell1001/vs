from collections import Counter, defaultdict
from math import isclose
from .passengers import direct_duration

def validate(solution, net, trips, p, required=None):
    if solution.metadata.get('timetable'):
        from common.timetable import make_trips
        trips = make_trips(net, solution.metadata['timetable'], p, solution.metadata.get('operations', ()))
    errors = []
    from .service_limits import direct_drt_time, completion_limits, line_deadline, duty_in_horizon
    limits = completion_limits(trips or {})
    if solution.metadata.get('branch_base'):
        from .timing import validate_branch
        if not validate_branch(solution.metadata['timetable'], solution.metadata['branch_base'], p):
            errors.append('Timetable outside branch adjustment limits')
    if any((l.kind == 'fixed' and l.trip not in trips for d in solution.duties.values() for l in d.legs)):
        errors.append('Unscheduled independent fixed-line service')
    people = {x.id: x for x in net.passengers}
    if solution.metadata.get('timetable'):
        from .base import build_base
        from .routing import construct_duties, Infeasible
        nominal = make_trips(net, solution.metadata['timetable'], p)
        try:
            reconstructed, actions = construct_duties(net, nominal, build_base(net, nominal, p), solution.excursions, p, relaxed=True, with_operations=True)



            scheduled = make_trips(net, solution.metadata['timetable'], p, actions)
            reconstructed = construct_duties(net, scheduled, build_base(net, scheduled, p), solution.excursions, p, operation_schedule=actions)
            for key, duty in solution.duties.items():
                expected_solo = [leg for leg in reconstructed[key].legs if leg.kind == 'solo'] if key in reconstructed else []
                actual_solo = [leg for leg in duty.legs if leg.kind == 'solo']
                if len(actual_solo) != len(expected_solo) or any(
                    (a.start, a.end, a.kind, a.trip, a.line) != (b.start, b.end, b.kind, b.trip, b.line)
                    or any(not isclose(x, y, rel_tol=0, abs_tol=1e-6) for x, y in
                           ((a.departure, b.departure), (a.arrival, b.arrival), (a.distance, b.distance)))
                    for a, b in zip(actual_solo, expected_solo)
                ):
                    errors.append('Independent continuation differs from scheduling decisions')
            if actions != {tuple(x) for x in solution.metadata.get('operations', ())}:
                errors.append('Operation dwell mask does not match actual scheduling decisions')
        except (Infeasible, KeyError, IndexError) as error:
            errors.append(f'Invalid module scheduling decisions: {error}')
    from .virtual import available_trips
    targets = available_trips(solution.duties, trips)
    expected = {x.id for x in net.passengers if not x.kind} | (set(required) if required is not None else {x.id for x in net.passengers if x.kind})
    if set(solution.services) != expected:
        errors.append('Passenger service set mismatch')
    pickup_counts = Counter((v.node for d in solution.duties.values() for v in d.visits if v.kind == 'pickup'))
    drop_counts = Counter((v.node for d in solution.duties.values() for v in d.visits if v.kind == 'dropoff'))
    for pid in expected:
        person = people[pid]
        if person.kind and (pickup_counts[person.origin] != 1 or drop_counts[person.dest] != 1):
            errors.append(f'{pid}: pickup/delivery must each occur exactly once')
    from .routing import cross_line_spans, Infeasible
    try:
        crossings = cross_line_spans(solution.duties, trips, solution.excursions)
    except Infeasible as error:
        errors.append(str(error))
        crossings = {}
    load = {k: [0] * len(d.legs) for k, d in solution.duties.items()}
    formations = Counter()
    vehicle_paths = defaultdict(list)
    terminals = {stops[0] for stops in net.lines.values()}
    for key, d in solution.duties.items():
        if len(d.visits) != len(d.legs) + 1:
            errors.append(f'{key}: invalid visit/leg cardinality')
            continue
        vehicle_paths[d.vehicle].append(d)
        if not duty_in_horizon(d, limits):
            errors.append(f'{key}: module task exceeds direction last-trip arrival')
        for v in d.visits:
            if v.kind in ('join', 'stored_join'):
                if v.arrival + p.operation_time > v.departure + 1e-07:
                    errors.append('Missed coupling operation window')
                if v.kind == 'stored_join' and v.node not in net.depots:
                    errors.append('Storage away from interchange')
                canonical = next((w for w in targets[v.trip].visits if w.node == v.node), None) if v.trip in targets else None
                if v.kind == 'join' and canonical is not None and (canonical.departure - p.operation_time - v.arrival > p.max_coupling_wait + 1e-07):
                    errors.append('Coupling wait exceeds configured maximum')
            if v.trip in targets and v.kind != 'depot':
                canonical = next((w for w in targets[v.trip].visits if w.node == v.node), None)
                if canonical is None or abs(canonical.departure - v.departure) > 1e-06:
                    errors.append('Module fails to follow fixed trip departure time')
                elif v.kind in ('fixed', 'solo') and abs(canonical.arrival - v.arrival) > 1e-06:
                    errors.append('Module fails to follow fixed trip arrival time')
        for i, leg in enumerate(d.legs):
            a, b = d.visits[i:i + 2]
            if (leg.start, leg.end) != (a.node, b.node):
                errors.append(f'{key}: discontinuous geometry')
            if not (a.arrival <= a.departure + 1e-07 and abs(a.departure - leg.departure) < 1e-06 and (abs(leg.arrival - b.arrival) < 1e-06) and (leg.arrival >= leg.departure - 1e-07)):
                errors.append(f'{key}: discontinuous time')
            minimum = net.road(leg.start, leg.end)[1] if leg.kind in ('fixed', 'solo', 'empty') and (leg.start, leg.end) in net.edges else net.distance(leg.start, leg.end) / p.speed * 60
            if leg.arrival - leg.departure < minimum - 1e-06:
                errors.append(f'{key}: impossible travel speed')
            if leg.kind == 'solo':
                edge = net.edges.get((leg.start, leg.end))
                if edge is None or abs(leg.distance - edge[0]) > 1e-06 or abs(leg.arrival - leg.departure - edge[1]) > 1e-06:
                    errors.append('Incorrect independent fixed-route arc')
            if leg.kind == 'fixed':
                formations[leg.trip, leg.start, leg.end, round(leg.departure, 7)] += 1
                edge = net.edges.get((leg.start, leg.end))
                if edge is None or abs(leg.distance - edge[0]) > 1e-06 or abs(leg.arrival - leg.departure - edge[1]) > 1e-06:
                    errors.append('Incorrect fixed-route distance or travel time')
            elif leg.kind == 'drt' and abs(leg.distance - net.distance(leg.start, leg.end)) > 1e-06:
                errors.append('Incorrect DRT arc distance')
    for trip in trips.values():
        for leg in trip.legs:
            n = formations[leg.trip, leg.start, leg.end, round(leg.departure, 7)]
            if not 1 <= n <= p.max_formation:
                errors.append(f'{trip.id}: formation {n}')
    if any((n > p.max_formation for n in formations.values())):
        errors.append('Oversized backup formation')
    for pid, s in solution.services.items():
        if pid not in people:
            errors.append('Unknown passenger')
            continue
        person = people[pid]
        if not s.rides or len(s.rides) > (2 if person.kind else 1):
            errors.append(f'{pid}: transfer count')
            continue
        valid = True
        for r in s.rides:
            if r.duty not in solution.duties or not 0 <= r.board <= r.alight < len(solution.duties[r.duty].visits):
                errors.append(f'{pid}: invalid ride indices')
                valid = False
                continue
            for i in range(r.board, r.alight):
                load[r.duty][i] += 1
        if not valid:
            continue
        first = s.rides[0]
        last = s.rides[-1]
        board = solution.duties[first.duty].visits[first.board]
        alight = solution.duties[last.duty].visits[last.alight]
        if (board.node, alight.node) != (person.origin, person.dest):
            errors.append(f'{pid}: wrong OD')
        actual_pickup = board.arrival if person.kind else board.departure
        if abs(actual_pickup - s.pickup) > 1e-06 or abs(alight.arrival - s.dropoff) > 1e-06:
            errors.append(f'{pid}: inconsistent service times')
        if s.pickup < person.earliest - 1e-07 or s.dropoff < s.pickup - 1e-07:
            errors.append(f'{pid}: pickup precedence')
        if s.dropoff > line_deadline(alight.line or person.line, limits) + 1e-7:
            errors.append(f'{pid}: service exceeds direction last-trip arrival')
        if person.kind:
            extra = max(0.0, s.dropoff - s.pickup - direct_drt_time(net, person, p))
            if extra > p.max_drt_detour + 1e-7 or abs(extra - s.detour) > 1e-6:
                errors.append(f'{pid}: excess/wrong DRT extra ride time')
        if person.kind and s.pickup > person.latest + 1e-07:
            errors.append(f'{pid}: pickup window')
        for a, b in zip(s.rides, s.rides[1:]):
            da, db = (solution.duties[a.duty], solution.duties[b.duty])
            va, vb = (da.visits[a.alight], db.visits[b.board])
            if va.kind == 'stored_join':
                errors.append('DRT donor carries passenger through warehouse storage')
            if not va.trip or (va.node, va.trip, va.departure) != (vb.node, vb.trip, vb.departure):
                errors.append(f'{pid}: uncoupled transfer')
            common = False
            for offset in (-1, 0):
                i, j = (a.alight + offset, b.board + offset)
                if 0 <= i < len(da.legs) and 0 <= j < len(db.legs):
                    x, y = (da.legs[i], db.legs[j])
                    if x.kind == y.kind == 'fixed' and (x.trip, x.start, x.end, x.departure) == (y.trip, y.start, y.end, y.departure):
                        common = True
            if not common:
                errors.append(f'{pid}: no shared coupled segment')
        if not person.kind:
            r = s.rides[0]
            duty = solution.duties[r.duty]
            if any((leg.kind == 'empty' for leg in duty.legs[r.board:r.alight])):
                errors.append('Ordinary passenger on empty reposition')
            if any((r.board < end and r.alight > start for start, end in crossings.get(r.duty, []))):
                errors.append('Ordinary passenger aboard during cross-line dispatch')
            route = net.lines[person.line]
            a, b = (route.index(person.origin), route.index(person.dest))
            direct = direct_duration(net, person.line, person.origin, person.dest, p)
            actual = max(0, s.dropoff - s.pickup - direct)
            offroute = sum((net.distance(leg.start, leg.end) / p.speed * 60 if leg.kind == 'drt' else net.road(leg.start, leg.end)[1] for leg in duty.legs[r.board:r.alight] if leg.kind not in ('fixed', 'solo') and leg.start != leg.end))
            if abs(offroute - s.offroute_minutes) > 1e-06:
                errors.append(f'{pid}: incorrect Eq. (4) off-route time')
            if actual > p.max_detour + 1e-07 or abs(actual - s.detour) > 1e-06:
                errors.append(f'{pid}: excess/wrong detour')
            if any((v.line != person.line for v in duty.visits[r.board + 1:r.alight + 1])) or any((v.kind in ('stored_join', 'depot') for v in duty.visits[r.board + 1:r.alight])):
                errors.append(f'{pid}: ordinary rider crosses lines or enters storage')
    if any((n > p.capacity for loads in load.values() for n in loads)):
        errors.append('Module passenger capacity exceeded')
    for key, duty in solution.duties.items():
        for i, v in enumerate(duty.visits):
            if v.kind in ('stored_join', 'depot') and any((r.duty == key and r.board < i < r.alight for service in solution.services.values() for r in service.rides)):
                errors.append('Occupied warehouse storage')
    moves = solution.metadata.get('rebalance_moves', [])
    if solution.metadata.get('fleet_size') != len(vehicle_paths):
        errors.append('Incorrect activated fleet count')
    for move in moves:
        edge = net.edges.get((move['start'], move['end']))
        if edge is None or move['arrival'] - move['departure'] < edge[1] - 1e-07 or abs(move['distance'] - edge[0]) > 1e-07:
            errors.append('Invalid physical empty-rebalance arc')
    actual_storage = []
    consumed_moves = set()
    for duty in solution.duties.values():
        for v in duty.visits:
            if v.kind == 'stored_join':
                actual_storage.append((v.node, duty.vehicle, v.arrival + p.operation_time, v.departure - p.operation_time))
        for v in duty.visits[:-1]:
            if v.kind == 'depot' and v is not duty.visits[0] and v.departure > v.arrival + p.operation_time + 1e-07:
                actual_storage.append((v.node, duty.vehicle, v.arrival + p.operation_time, v.departure))
    for vehicle, chain in vehicle_paths.items():
        chain.sort(key=lambda d: (d.visits[0].departure, bool(d.legs), d.id))
        if chain[0].visits[0].node not in terminals | set(net.depots):
            errors.append('Module activated away from a line terminal')
        if chain[-1].visits[-1].node not in terminals | set(net.depots):
            errors.append('Module has no valid final terminal/storage location')
        last = chain[-1].visits[-1]
        if last.kind == 'depot':
            actual_storage.append((last.node, vehicle, last.departure, None))
        first = chain[0].visits[0]
        births = [a for a in solution.metadata.get('activations', ()) if a['vehicle'] == vehicle]
        if len(births) != 1 or births[0]['node'] != first.node or abs(births[0]['time'] - first.departure) > 1e-06:
            errors.append('Incorrect physical module activation record')
        expected_origin = 'reserve' if chain[0].id.startswith('reserve:') else 'terminal'
        if births and births[0].get('origin', 'terminal') != expected_origin:
            errors.append('Incorrect activation origin')
        if expected_origin == 'reserve' and first.node not in net.depots:
            errors.append('Reserve vehicle activated away from warehouse')
        for a, b in zip(chain, chain[1:]):
            end, start = (a.visits[-1], b.visits[0])
            if net.line_families.get(end.line, end.line) != net.line_families.get(start.line, start.line) and not (end.node in terminals and start.node in terminals) and end.kind != 'depot' and start.kind != 'depot' and (end.node not in net.depots or start.node not in net.depots):
                errors.append('Cross-line reuse outside interchange')
            if end.departure + p.operation_time > start.departure + 1e-07:
                errors.append('Overlapping module duties')
            ready = end.departure
            if end.node != start.node:
                relevant = sorted([m for m in moves if m['vehicle'] == vehicle and m['departure'] >= end.departure - 1e-07 and (m['arrival'] <= start.departure + 1e-07)], key=lambda m: m['departure'])
                node = end.node
                time = end.departure
                for m in relevant:
                    consumed_moves.add(id(m))
                    if m['start'] != node or m['departure'] < time - 1e-07:
                        errors.append('Discontinuous empty rebalance')
                    node, time = (m['end'], m['arrival'])
                if node != start.node:
                    errors.append('Module teleportation between duties')
                ready = time
                if time + p.operation_time > start.departure + 1e-07:
                    errors.append('Empty rebalance misses coupling readiness')
            if start.node in net.depots and ready < start.departure - 1e-07:
                actual_storage.append((start.node, vehicle, ready, start.departure))
    if any((id(move) not in consumed_moves for move in moves)):
        errors.append('Unmatched physical empty-rebalance arc')

    def normalized(spans):
        return sorted(((int(node), int(vehicle), round(a, 6), None if b is None else round(b, 6)) for node, vehicle, a, b in spans if b is None or b > a + 1e-07))
    declared = [(s['node'], s['vehicle'], s['in'], s['out']) for s in solution.metadata.get('storage', ())]
    if normalized(actual_storage) != normalized(declared):
        errors.append('Warehouse records do not match physical module paths')
    from .resources import inventory
    from .routing import Infeasible
    try:
        actual = inventory(solution.metadata.get('storage', ()), net, p, solution.metadata.get('activations', ()))
        reported = solution.metadata.get('inventory', {})
        if any((actual[node] != reported.get(node, reported.get(str(node))) for node in actual)):
            errors.append('Incorrect warehouse inventory event report')
    except Infeasible as error:
        errors.append(str(error))
    return errors
