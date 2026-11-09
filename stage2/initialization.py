from dataclasses import replace
from common.model import Excursion
from common.timetable import make_trips
from .routing import Infeasible

def sequence(net, people, events, origin, p):
    node = origin.node
    time = origin.departure
    distance = waiting = 0.0
    onboard = set()
    for pid, pickup in events:
        q = people[pid]
        target = q.origin if pickup else q.dest
        d = net.distance(node, target)
        distance += d
        time += d / p.speed * 60
        if pickup:
            time = max(time, q.earliest)
            if time > q.latest + 1e-07:
                return None
            onboard.add(pid)
            waiting += time - q.earliest
        else:
            if pid not in onboard:
                return None
            onboard.remove(pid)
        if len(onboard) > p.capacity:
            return None
        node = target
        time += p.stop_time
    if onboard:
        return None
    return (distance, waiting, node, time)

def operations(net, trips, plan):
    result = {(e.split_trip, e.split_node) for e in plan}
    for e in plan:
        if e.target in trips:
            result.add((e.target, trips[e.target].visits[e.join].node))
    return result

def interval(net, people, events, origin, p):
    from .synchronization import service_times
    onboard = set()
    for pid, pickup in events:
        if pickup:
            if pid in onboard:
                return None
            onboard.add(pid)
        elif pid not in onboard:
            return None
        else:
            onboard.remove(pid)
        if len(onboard) > p.capacity:
            return None
    if onboard:
        return None
    try:
        times = service_times(net, people, events, origin.node, origin.departure, None, None, p)
    except Infeasible:
        return None
    if not events or not any(pickup for _, pickup in events):
        return None
    nodes = [people[pid].origin if pickup else people[pid].dest for pid, pickup in events]
    distance = net.distance(origin.node, nodes[0]) + sum(net.distance(a, b) for a, b in zip(nodes, nodes[1:]))
    waiting = sum(times[i] - people[pid].earliest for i, (pid, pickup) in enumerate(events) if pickup)
    last = max(i for i, (_, pickup) in enumerate(events) if pickup)
    pid, _ = events[last]
    latest = people[pid].latest + p.stop_time
    node = people[pid].origin
    for q, pickup in events[last + 1:]:
        target = people[q].origin if pickup else people[q].dest
        latest += net.distance(node, target) / p.speed * 60 + p.stop_time
        node = target
    return distance, waiting, nodes[-1], times[-1] + p.stop_time, latest


def cached_trips(decoder, mask):
    mask = frozenset(mask)
    cache = getattr(decoder, 'draft_cache', None)
    if cache is None:
        cache = decoder.draft_cache = {}
    if mask not in cache:
        cache[mask] = (make_trips(decoder.net, decoder.timetable, decoder.p, mask), {})
        if len(cache) > 256:
            cache.pop(next(iter(cache)))
    return cache[mask]

def protect_ordinary(decoder, trips, duties, plan):
    from bisect import bisect_left
    from collections import Counter
    from .passengers import direct_duration
    net, p = decoder.net, decoder.p
    people = getattr(decoder, 'ordinary_construction_people', None)
    if people is None:
        people = {line: [] for line in net.lines}
        for q in net.passengers:
            if not q.kind:
                route = net.lines[q.line]
                people[q.line].append((route.index(q.origin), q.earliest, q.id, route.index(q.dest)))
        for rows in people.values():
            rows.sort()
        decoder.ordinary_construction_people = people
    cache = getattr(decoder, 'ordinary_construction_cache', None)
    if cache is None:
        cache = decoder.ordinary_construction_cache = {}
    forced = {e.source for e in plan}
    counts = Counter((leg.trip, leg.start, leg.end, round(leg.departure, 7)) for key in forced for leg in duties[key].legs if leg.kind == 'fixed')
    removed = Counter(e.split_trip for e in plan)
    for line, route in net.lines.items():
        rows = sorted((trip for trip in trips.values() if trip.line == line), key=lambda t: int(t.id.rsplit(':', 1)[1]))
        free = [min([p.max_formation - removed[t.id]] + [p.max_formation - counts[t.id, leg.start, leg.end, round(leg.departure, 7)] for leg in t.legs]) for t in rows]
        signature = (line, tuple(free), tuple(tuple((v.arrival, v.departure) for v in t.visits) for t in rows))
        if signature in cache:
            if not cache[signature]:
                raise Infeasible('Construction cannot preserve complete ordinary boarding')
            continue
        loads = [[0] * len(t.legs) for t in rows]
        departures = [[t.visits[i].departure for t in rows] for i in range(len(route))]
        feasible = True
        for a, earliest, pid, b in people[line]:
            canonical = direct_duration(net, line, route[a], route[b], p)
            k = bisect_left(departures[a], earliest - 1e-7)
            while k < len(rows):
                if (rows[k].visits[b].arrival - rows[k].visits[a].departure - canonical <= p.max_detour + 1e-7
                        and all(loads[k][i] < free[k] * p.capacity for i in range(a, b))):
                    break
                k += 1
            if k == len(rows):
                feasible = False
                break
            for i in range(a, b):
                loads[k][i] += 1
        if len(cache) >= 512:
            cache.pop(next(iter(cache)))
        cache[signature] = feasible
        if not feasible:
            raise Infeasible('Construction cannot preserve complete ordinary boarding')

def continuation_check(decoder, plan):
    from .base import build_base
    from .routing import construct_duties
    from .initial_capacity import CapacityGuard
    from .initial_inventory import check_inventory
    net, p = decoder.net, decoder.p
    mask = operations(net, decoder.nominal_trips, plan)
    trips, demand_cache = cached_trips(decoder, mask)
    sources = {e.source for e in plan}
    duties = construct_duties(net, trips, build_base(net, trips, p, sources), plan, p, operation_schedule=mask)
    for key in sources:
        load = 0
        for visit in duties[key].visits:
            load += 1 if visit.kind == 'pickup' else -1 if visit.kind == 'dropoff' else 0
            if not 0 <= load <= p.capacity:
                raise Infeasible('DRT capacity')
    from .service_limits import completion_limits, duty_in_horizon
    limits = completion_limits(trips)
    if any(not duty_in_horizon(duties[key], limits) for key in sources):
        raise Infeasible('Task completes after its direction last-trip arrival')
    check_inventory(duties, plan, net, p)
    protect_ordinary(decoder, trips, duties, plan)
    guard = getattr(decoder, 'initial_capacity_guard', None)
    if guard is None:
        guard = decoder.initial_capacity_guard = CapacityGuard(net, p)
    guard.demand_cache = demand_cache
    guard.check(trips, duties, plan)
    return trips

def draft_check(decoder, plan):
    if plan:
        continuation_check(decoder, plan)
    if any(e.target not in decoder.nominal_trips for e in plan):
        return cached_trips(decoder, operations(decoder.net, decoder.nominal_trips, plan))[0]
    import numpy as np
    from collections import Counter
    from bisect import bisect_left
    net, p = (decoder.net, decoder.p)
    nominal = decoder.nominal_trips
    mask = frozenset(operations(net, nominal, plan))
    trips, demand_cache = cached_trips(decoder, mask)
    joins = Counter()
    leaves = Counter()
    forced = Counter()
    for e in plan:
        if e.target not in trips:
            raise Infeasible('Return must choose a scheduled trip')
        source = trips[e.split_trip]
        target = trips[e.target]
        a = source.visits[e.split]
        b = target.visits[e.join]
        if not 0 <= e.split < len(source.visits) - 1 or not 0 <= e.join < len(target.visits) - 1:
            raise Infeasible('Split/join must use an intermediate station')
        if source.line != target.line and (a.node not in net.depots or b.node not in net.depots):
            raise Infeasible('Cross-line split/join must use an interchange')
        if source.line == target.line and (int(target.id.rsplit(':', 1)[1]) <= int(source.id.rsplit(':', 1)[1]) or net.lines[target.line].index(b.node) <= e.split):
            raise Infeasible('Same-line return must join a later downstream scheduled trip')
        profile = interval(net, decoder.people, e.events, a, p)
        if profile is None:
            raise Infeasible('Pickup sequence/window/capacity')
        _, _, node, earliest, latest = profile
        travel = net.distance(node, b.node) / p.speed * 60
        couple = b.departure - p.operation_time
        if earliest + travel > couple + 1e-07 or latest + travel < couple - p.max_coupling_wait - 1e-07:
            raise Infeasible('Return coupling window')
        leaves[source.id, e.split] += 1
        joins[target.id, e.join] += 1
        forced[source.id] += 1
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

def construct(decoder, rule, rng, progress=print, initial=(), pending=None):
    from .budget import check
    net, p = (decoder.net, decoder.p)
    people = decoder.people
    initial_mode = pending is None
    pending_order = list(pending) if pending is not None else []
    plan = list(initial)
    checks = 0
    budget = p.initial_checks if initial_mode else p.repair_checks
    if not initial_mode:
        pending = set(pending)
        from .operators import remove
        while plan:
            trips, _ = cached_trips(decoder, operations(net, decoder.nominal_trips, plan))
            invalid = None
            for e in plan:
                a = trips[e.split_trip].visits[e.split]
                b = trips[e.target].visits[e.join] if e.target in trips else None
                profile = interval(net, people, e.events, a, p)
                if profile is None:
                    invalid = e
                    break
                _, _, node, lo, hi = profile
                travel = net.distance(node, b.node) / p.speed * 60 if b is not None else 0
                couple = b.departure - p.operation_time if b is not None else None
                if b is not None and (lo + travel > couple + 1e-07 or hi + travel < couple - p.max_coupling_wait - 1e-07):
                    invalid = e
                    break
            if invalid is None:
                break
            old = {q for e in plan for q, _ in e.events}
            plan = list(remove(plan, {q for q, _ in invalid.events}))
            pending.update(old - {q for e in plan for q, _ in e.events})
    requests = sorted((q for q in net.passengers if q.kind and (pending is None or q.id in pending)), key=lambda q: (q.latest, q.id))
    if not initial_mode and rule == 0:
        order = {pid: i for i, pid in enumerate(pending_order)}
        requests.sort(key=lambda q: (order.get(q.id, len(order)), q.latest, q.id))
    elif not initial_mode and rule == 1:
        requests.sort(key=lambda q: (q.latest, q.latest - q.earliest, q.id))

    def offers(person, independent=False):
        nonlocal checks
        nominal = decoder.nominal_trips

        def endings(e, base, pos):
            nonlocal checks
            ops = operations(net, nominal, base)
            trips, _ = cached_trips(decoder, ops)
            source = trips[e.split_trip]
            origin = source.visits[e.split]
            if (source.id, origin.node) not in ops:
                origin = replace(origin, departure=origin.departure + p.operation_time)
            profile = interval(net, people, e.events, origin, p)
            if profile is None:
                return
            distance, waiting, node, earliest, latest = profile
            from .continuation_cost import cost
            position = net.lines[source.line].index(origin.node)
            choices = []
            if independent and decoder.allow_solo:
                for j in range(position + 1, len(net.lines[source.line])):
                    target = f'backup:{source.line}'
                    tail = sum(net.road(a, b)[0] for a, b in zip(net.lines[source.line][j:], net.lines[source.line][j + 1:]))
                    score = net.distance(node, net.lines[source.line][j]) + p.following_ratio * tail
                    choices.append((p.operating_cost * score, target, j))
            for target in (() if independent else trips.values()):
                if target.id == source.id:
                    continue
                for j, visit in enumerate(target.visits[:-1]):
                    if target.line == source.line:
                        if int(target.id.rsplit(':', 1)[1]) <= int(source.id.rsplit(':', 1)[1]) or j <= position:
                            continue
                    elif origin.node not in net.depots or visit.node not in net.depots:
                        continue
                    couple = visit.departure + (p.operation_time if (target.id, visit.node) not in ops else 0) - p.operation_time
                    travel = net.distance(node, visit.node) / p.speed * 60
                    if earliest + travel > couple + 1e-07 or latest + travel < couple - p.max_coupling_wait - 1e-07:
                        continue
                    choices.append((cost(net, p, node, target.id, j, trips), target.id, j))
            from .service_limits import completion_limits, line_deadline
            horizon = line_deadline(source.line, completion_limits(trips))
            for _, target, j in sorted(choices):
                if independent:
                    route = net.lines[source.line]
                    remaining = sum(net.road(a, b)[1] for a, b in zip(route[j:], route[j + 1:])) + max(0, len(route) - j - 1) * p.stop_time
                    if earliest + net.distance(node, route[j]) / p.speed * 60 + remaining > horizon + 1e-7:
                        continue
                check(decoder)
                checks += 1
                decoder.last_construction = {'orders_inserted': index, 'orders_total': len(requests), 'insertion_checks': checks, 'local_backtracks': backtracks}
                if checks > budget:
                    raise Infeasible(f'Insertion-check budget exhausted at DRT {person.id}; {index}/{len(requests)} inserted')
                end = replace(e, target=target, join=j)
                trial = list(base)
                if pos is None:
                    trial.append(end)
                else:
                    trial.insert(pos, end)
                try:
                    draft_check(decoder, trial)
                except Infeasible as error:
                    decoder.last_rejection = str(error)
                    continue
                yield tuple(trial)
        for pos, e in enumerate(plan):
            if not e.events:
                continue
            base = plan[:pos] + plan[pos + 1:]
            for a in range(len(e.events)):
                if a and sum(1 if pickup else -1 for _, pickup in e.events[:a]) == 0:
                    continue
                for b in range(a + 1, len(e.events) + 2):
                    check(decoder)
                    events = list(e.events)
                    events.insert(a, (person.id, True))
                    events.insert(b, (person.id, False))
                    yield from endings(replace(e, events=tuple(events)), base, pos)
        ops = operations(net, nominal, plan)
        trips, _ = cached_trips(decoder, ops)
        occupied = {}
        for e in plan:
            occupied.setdefault(e.split_trip, set()).add(int(e.source.rsplit('m', 1)[1]))
        sources = []
        for trip in trips.values():
            used = occupied.get(trip.id, set())
            slot = next((m for m in range(1, p.max_formation) if m not in used), None)
            if slot is None:
                continue
            for i, v in enumerate(trip.visits[:-1]):
                start = v.departure + (p.operation_time if (trip.id, v.node) not in ops else 0)
                pickup = max(person.earliest, start + net.distance(v.node, person.origin) / p.speed * 60)
                if pickup <= person.latest + 1e-07:
                    sources.append((pickup - person.earliest, max(0.0, person.earliest - start - net.distance(v.node, person.origin) / p.speed * 60), net.distance(v.node, person.origin), trip.id, i, slot))
        for _, _, _, trip, i, slot in sorted(sources):
            v = trips[trip].visits[i]
            e = Excursion(f'{trip}/m{slot}', i, '', 0, ((person.id, True), (person.id, False)), trip, v.node)
            yield from endings(e, plan, None)
    index = 0
    backtracks = 0
    history = []
    blocked_orders = {}
    while index < len(requests):
        decoder.last_construction = {'orders_inserted': index, 'orders_total': len(requests), 'insertion_checks': checks, 'local_backtracks': backtracks}
        q = requests[index]
        check(decoder)
        chosen = next(offers(q), None)
        if chosen is None and decoder.allow_solo:
            chosen = next(offers(q, independent=True), None)
        if chosen is None:
            if initial_mode and index and (backtracks < p.initial_backtracks):
                backtracks += 1
                attempts = blocked_orders.get(q.id, 0) + 1
                blocked_orders[q.id] = attempts
                width = min(index, 4 * 2 ** min(attempts - 1, 4))
                anchor = index - width
                plan = list(history[anchor])
                history = history[:anchor]
                blocked = requests.pop(index)
                requests.insert(anchor, blocked)
                index = anchor
                progress(f'  Reorder {width} recent orders; prioritize DRT {q.id}; repair {backtracks}/{p.initial_backtracks}')
                continue
            raise Infeasible(f'Constructive insertion failed for DRT {q.id}; {index} requests inserted; {checks} checks, {backtracks} local backtracks')
        if initial_mode:
            history.append(tuple(plan))
        plan = list(chosen)
        index += 1
        if index % 10 == 0:
            progress(f'  Constructed {index}/{len(requests)} DRT routes')
    decoder.last_construction = {'insertion_checks': checks, 'local_backtracks': backtracks, 'orders': len(requests)}
    decoder.initialization_stats = {'strategy': 'scheduled_first_feasible' if initial_mode else ('greedy_first_feasible', 'urgent_first_feasible', 'synchronization_first_feasible')[rule], 'path_checks': checks, 'backtracks': backtracks, 'inserted': len(requests), 'requests': len(requests), 'request_order': [q.id for q in requests]}
    return tuple(plan)
