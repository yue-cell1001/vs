from dataclasses import replace
from math import inf
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
    result = sequence(net, people, events, origin, p)
    if result is None:
        return None
    last = max((i for i, (_, pickup) in enumerate(events) if pickup), default=-1)
    if last < 0:
        return None
    pid, _ = events[last]
    latest = people[pid].latest + p.stop_time
    node = people[pid].origin
    for q, pickup in events[last + 1:]:
        target = people[q].origin if pickup else people[q].dest
        latest += net.distance(node, target) / p.speed * 60 + p.stop_time
        node = target
    return (*result, latest)

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

def draft_check(decoder, plan):
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
        if e.split == 0 or e.join == 0 or e.join >= len(target.visits) - 1:
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
    from itertools import islice
    from .budget import check
    net, p = (decoder.net, decoder.p)
    people = decoder.people
    initial_mode = pending is None
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
                b = trips[e.target].visits[e.join]
                profile = interval(net, people, e.events, a, p)
                if profile is None:
                    invalid = e
                    break
                _, _, node, lo, hi = profile
                travel = net.distance(node, b.node) / p.speed * 60
                couple = b.departure - p.operation_time
                if lo + travel > couple + 1e-07 or hi + travel < couple - p.max_coupling_wait - 1e-07:
                    invalid = e
                    break
            if invalid is None:
                break
            old = {q for e in plan for q, _ in e.events}
            plan = list(remove(plan, {q for q, _ in invalid.events}))
            pending.update(old - {q for e in plan for q, _ in e.events})
    requests = sorted((q for q in net.passengers if q.kind and (pending is None or q.id in pending)), key=lambda q: (q.latest, q.id))

    def offers(person):
        nonlocal checks
        nominal = decoder.nominal_trips

        def endings(e, base, pos, old=None):
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
            from bisect import bisect_left, bisect_right
            mask = frozenset(ops)
            cache = getattr(decoder, 'return_index_cache', None)
            if cache is None:
                cache = decoder.return_index_cache = {}
            if mask not in cache:
                rows = {}
                for target in trips.values():
                    for j, v in enumerate(target.visits[:-1]):
                        if j:
                            couple = v.departure + (p.operation_time if (target.id, v.node) not in ops else 0) - p.operation_time
                            rows.setdefault(v.node, []).append((couple, target.id, j))
                indexed = {}
                for station, values in rows.items():
                    values.sort()
                    indexed[station] = ([v[0] for v in values], values)
                if len(cache) >= 256:
                    cache.pop(next(iter(cache)))
                cache[mask] = indexed
            choices = []
            for station, (times, values) in cache[mask].items():
                check(decoder)
                back = net.distance(node, station)
                travel = back / p.speed * 60
                lo = bisect_left(times, earliest + travel - 1e-7)
                hi = bisect_right(times, latest + travel + p.max_coupling_wait + 1e-7)
                for couple, target_id, j in values[lo:hi]:
                    target = trips[target_id]
                    if target.line != source.line and (origin.node not in net.depots or station not in net.depots):
                        continue
                    if target.id == source.id:
                        continue
                    if target.line == source.line and (int(target.id.rsplit(':', 1)[1]) <= int(source.id.rsplit(':', 1)[1]) or j <= e.split):
                        continue
                    choices.append((0 if station in net.depots else 1, couple - earliest - travel, station, target.id, j, distance + back, waiting))
            for _, _, _, target, j, distance, waiting in sorted(choices):
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
                except Infeasible:
                    continue
                baseline_distance, baseline_wait = old or (0.0, 0.0)
                yield ((0.0 if old is not None else p.purchase_cost) + p.operating_cost * (distance - baseline_distance), waiting - baseline_wait, tuple(trial))
        for pos, e in enumerate(plan):
            if not e.events:
                continue
            base = plan[:pos] + plan[pos + 1:]
            old_trips, _ = cached_trips(decoder, operations(net, nominal, plan))
            previous = sequence(net, people, e.events, old_trips[e.split_trip].visits[e.split], p)
            old = (previous[0] + net.distance(previous[2], old_trips[e.target].visits[e.join].node), previous[1]) if previous else (0.0, 0.0)
            for a in range(len(e.events) + 1):
                for b in range(a + 1, len(e.events) + 2):
                    check(decoder)
                    events = list(e.events)
                    events.insert(a, (person.id, True))
                    events.insert(b, (person.id, False))
                    yield from endings(replace(e, events=tuple(events)), base, pos, old)
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
                if i == 0:
                    continue
                start = v.departure + (p.operation_time if (trip.id, v.node) not in ops else 0)
                pickup = max(person.earliest, start + net.distance(v.node, person.origin) / p.speed * 60)
                if pickup <= person.latest + 1e-07:
                    sources.append((0 if v.node in net.depots else 1, pickup - person.earliest, net.distance(v.node, person.origin), trip.id, i, slot))
        for _, _, _, trip, i, slot in sorted(sources):
            v = trips[trip].visits[i]
            e = Excursion(f'{trip}/m{slot}', i, '', 0, ((person.id, True), (person.id, False)), trip, v.node)
            yield from endings(e, plan, None)
    index = 0
    backtracks = 0
    history = []
    skips = {}
    blocked_orders = {}
    while index < len(requests):
        decoder.last_construction = {'orders_inserted': index, 'orders_total': len(requests), 'insertion_checks': checks, 'local_backtracks': backtracks}
        q = requests[index]
        check(decoder)
        count = 1 if initial_mode else 6
        if not initial_mode and rule == 1:
            priorities = []
            scale = getattr(decoder, 'scales', (1.0, 1.0))
            for position in range(index, len(requests)):
                alternatives = list(islice(offers(requests[position]), 6))
                nd = [c for c in alternatives if not any((all((a <= b for a, b in zip(o[:2], c[:2]))) and any((a < b for a, b in zip(o[:2], c[:2]))) for o in alternatives))]
                nd.sort(key=lambda c: max((max(0.0, c[i]) / scale[i] for i in (0, 1))))
                scores = [max((max(0.0, c[i]) / scale[i] for i in (0, 1))) for c in nd]
                regret = inf if len(scores) <= 1 else scores[1] - scores[0]
                priorities.append((regret, -position, position, nd))
            _, _, position, candidates = max(priorities, key=lambda x: x[:2])
            requests[index], requests[position] = (requests[position], requests[index])
            q = requests[index]
        else:
            candidates = list(islice(offers(q), skips.get(index, 0), skips.get(index, 0) + count))
        if not candidates:
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
                skips.clear()
                index = anchor
                progress(f'  Reorder {width} recent orders; prioritize DRT {q.id}; repair {backtracks}/{p.initial_backtracks}')
                continue
            raise Infeasible(f'Constructive insertion failed for DRT {q.id}; {index} requests inserted; {checks} checks, {backtracks} local backtracks')
        if initial_mode:
            chosen = candidates[0]
        else:
            nd = [c for c in candidates if not any((all((a <= b for a, b in zip(other[:2], c[:2]))) and any((a < b for a, b in zip(other[:2], c[:2]))) for other in candidates))]
            scale = getattr(decoder, 'scales', (1.0, 1.0))
            if getattr(decoder, 'sample_repair_front', False):
                groups = {}
                for candidate in nd:
                    groups.setdefault(candidate[:2], []).append(candidate)
                chosen = rng.choice(groups[rng.choice(list(groups))])
            else:
                chosen = min(nd, key=lambda c: max((max(0.0, c[i]) / scale[i] for i in (0, 1))))
        history.append(tuple(plan))
        plan = list(chosen[2])
        index += 1
        if index % 10 == 0:
            progress(f'  Constructed {index}/{len(requests)} DRT routes')
    decoder.last_construction = {'insertion_checks': checks, 'local_backtracks': backtracks, 'orders': len(requests)}
    decoder.initialization_stats = {'strategy': 'scheduled_first_feasible' if initial_mode else ('pareto_greedy', 'regret2', 'synchronization')[rule], 'path_checks': checks, 'backtracks': backtracks, 'inserted': len(requests), 'requests': len(requests)}
    return tuple(plan)
