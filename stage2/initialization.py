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
    result = {(e.split_trip, e.split_node) for e in plan if e.split_trip != 'depot'}
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
    from .base import build_base
    from .routing import construct_duties
    from .initial_capacity import CapacityGuard
    from .initial_inventory import check_inventory
    net, p = decoder.net, decoder.p
    mask = operations(net, decoder.nominal_trips, plan)
    trips, demand_cache = cached_trips(decoder, mask)
    sources = {e.source for e in plan}
    built = construct_duties(net, trips, build_base(net, trips, p, sources), plan, p, operation_schedule=mask)
    for key in sources:
        pickups = {pid for e in plan if e.source == key for pid, pickup in e.events if pickup}
        dropoffs = {pid for e in plan if e.source == key for pid, pickup in e.events if not pickup}
        if pickups != dropoffs:
            continue
        load = 0
        for visit in built[key].visits:
            if visit.kind == 'pickup':
                load += 1
            elif visit.kind == 'dropoff':
                load -= 1
            if not 0 <= load <= p.capacity:
                raise Infeasible('DRT capacity')
    check_inventory(built, plan, net, p)
    guard = getattr(decoder, 'initial_capacity_guard', None)
    if guard is None:
        guard = decoder.initial_capacity_guard = CapacityGuard(net, p)
    guard.demand_cache = demand_cache
    guard.check(trips, built, plan)
    return trips

def construct(decoder, rule, rng, progress=print, initial=(), pending=None):
    if pending is not None:
        return _construct_preserved_repair(decoder, rule, rng, progress, initial, pending)
    from itertools import islice
    from .budget import check
    net, p = decoder.net, decoder.p
    people = decoder.people
    initial_mode = pending is None
    pending_order = list(pending) if pending is not None else []
    if initial_mode:
        plan = list(initial)
    else:
        from .repair_seed import prepare
        plan, pending = prepare(decoder, initial, pending)
    checks = 0
    budget = p.initial_checks if initial_mode else p.repair_checks
    requests = sorted((q for q in net.passengers if q.kind and (pending is None or q.id in pending)), key=lambda q: (q.latest, q.id))
    statistics = {'strategy': 'constructive_first_feasible' if initial_mode else 'constructive_repair', 'candidate_policy': 'ordered_without_shortlist_limits' if initial_mode else 'first_feasible_insertion', 'check_budget': budget, 'backtrack_limit': p.initial_backtracks if initial_mode else 0, 'allow_solo': decoder.allow_solo, 'requests': len(requests), 'inserted': 0, 'path_checks': 0}
    decoder.initialization_stats = statistics
    if not initial_mode and rule == 0:
        order = {pid: index for index, pid in enumerate(pending_order)}
        requests.sort(key=lambda q: (order.get(q.id, len(order)), q.latest, q.id))

    def offers(person):
        nonlocal checks
        nominal = decoder.nominal_trips

        def endings(e, base, pos, old=None):
            nonlocal checks
            from .base import build_base
            from .routing import construct_duties, split_index
            ops = operations(net, nominal, base)
            trips, _ = cached_trips(decoder, ops)
            if e.split_trip == 'depot':
                try:
                    built = construct_duties(net, trips, build_base(net, trips, p, {item.source for item in base} | {e.source}), base, p, operation_schedule=ops)
                    origin = built[e.source].visits[split_index(built[e.source], e)]
                except Infeasible:
                    return
                if e.release_time is not None:
                    origin = replace(origin, departure=max(origin.departure, e.release_time))
            else:
                source = trips[e.split_trip]
                origin = source.visits[e.split]
                if (source.id, origin.node) not in ops:
                    origin = replace(origin, departure=origin.departure + p.operation_time)
            profile = interval(net, people, e.events, origin, p)
            if profile is None:
                return
            distance, waiting, node, earliest, latest = profile
            from .initial_returns import endings as return_endings
            position = net.lines[origin.line].index(origin.node) if origin.line in net.lines and origin.node in net.lines[origin.line] else -1
            for target, j in return_endings(decoder, trips, ops, origin, e, node, earliest, latest, position, net.distance):
                check(decoder)
                checks += 1
                decoder.last_construction = {'orders_inserted': index, 'orders_total': len(requests), 'insertion_checks': checks, 'local_backtracks': backtracks}
                statistics.update(inserted=index, path_checks=checks, backtracks=backtracks)
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
                    statistics['last_rejection'] = str(error)
                    continue
                end_node = trips[target].visits[j].node if target in trips else int(target.split(':', 1)[1]) if target.startswith('depot:') else net.lines[target.split(':', 1)[1]][j]
                route_distance = distance + net.distance(node, end_node)
                if target.startswith('backup:'):
                    route = net.lines[target.split(':', 1)[1]]
                    route_distance += sum(net.road(a, b)[0] for a, b in zip(route[j:], route[j + 1:]))
                baseline_distance, baseline_wait = old or (0.0, 0.0)
                yield ((0.0 if old is not None else p.purchase_cost) + p.operating_cost * (route_distance - baseline_distance), waiting - baseline_wait, tuple(trial))
        for pos, e in enumerate(plan):
            if not e.events:
                continue
            base = plan[:pos] + plan[pos + 1:]
            old = (0.0, 0.0)
            if initial_mode:
                old_trips, _ = cached_trips(decoder, operations(net, nominal, plan))
                if e.split_trip != 'depot':
                    old_origin = old_trips[e.split_trip].visits[e.split]
                else:
                    from .base import build_base
                    from .routing import construct_duties, split_index
                    old_ops = operations(net, nominal, plan)
                    prior = construct_duties(net, old_trips, build_base(net, old_trips, p, {item.source for item in plan}), plan[:pos], p, operation_schedule=old_ops)
                    old_origin = prior[e.source].visits[split_index(prior[e.source], e)]
                    if e.release_time is not None:
                        old_origin = replace(old_origin, departure=max(old_origin.departure, e.release_time))
                previous = sequence(net, people, e.events, old_origin, p)
                old_target = old_trips[e.target].visits[e.join].node if e.target in old_trips else int(e.target.split(':', 1)[1]) if e.target.startswith('depot:') else net.lines[e.target.split(':', 1)[1]][e.join]
                old = (previous[0] + net.distance(previous[2], old_target), previous[1]) if previous else (0.0, 0.0)
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
                start = v.departure + (p.operation_time if (trip.id, v.node) not in ops else 0)
                pickup = max(person.earliest, start + net.distance(v.node, person.origin) / p.speed * 60)
                if pickup <= person.latest + 1e-07:
                    sources.append((pickup - person.earliest, net.distance(v.node, person.origin), trip.id, i, slot))
        for _, _, trip, i, slot in sorted(sources):
            v = trips[trip].visits[i]
            e = Excursion(f'{trip}/m{slot}', i, '', 0, ((person.id, True), (person.id, False)), trip, v.node)
            yield from endings(e, plan, None)
        try:
            context = decoder.context(tuple(plan)) if plan else None
        except Infeasible:
            context = None
        if context is not None:
            reusable = {e.source for e in plan if context.duties[e.source].visits[-1].kind == 'depot'}
            for source in sorted(reusable, key=lambda key: (max(context.duties[key].visits[-1].departure + net.distance(context.duties[key].visits[-1].node, person.origin) / p.speed * 60, person.earliest), key)):
                duty = context.duties[source]
                end = duty.visits[-1]
                release = max(end.departure, person.earliest - net.distance(end.node, person.origin) / p.speed * 60)
                e = Excursion(source, len(duty.visits) - 1, '', 0, ((person.id, True), (person.id, False)), 'depot', end.node, release)
                yield from endings(e, plan, None)
        used = {e.source for e in plan}
        for depot in sorted(net.depots, key=lambda node: (net.distance(node, person.origin), node)):
            source = next((f'reserve:{depot}/m{slot}' for slot in range(p.depot_capacity) if f'reserve:{depot}/m{slot}' not in used), None)
            if source is None:
                continue
            release = max(p.start, person.earliest - net.distance(depot, person.origin) / p.speed * 60)
            e = Excursion(source, 0, '', 0, ((person.id, True), (person.id, False)), 'depot', depot, release)
            yield from endings(e, plan, None)
    index = 0
    backtracks = 0
    history = []
    skips = {}
    blocked_orders = {}
    if not initial_mode:
        for index, person in enumerate(requests):
            check(decoder)
            statistics.update(inserted=index, path_checks=checks)
            selected = next(offers(person), None)
            if selected is None:
                raise Infeasible(f'Constructive insertion failed for DRT {person.id}; {index} requests inserted; {checks} checks')
            plan = list(selected[2])
        statistics.update(inserted=len(requests), path_checks=checks, backtracks=0, request_order=[q.id for q in requests], ordinary_assignment_policy='reallocated_by_final_joint_model', draft_capacity_policy='residual_capacity_upper_bound')
        decoder.optimizer.initial_hint = None
        return tuple(plan)
    while index < len(requests):
        decoder.last_construction = {'orders_inserted': index, 'orders_total': len(requests), 'insertion_checks': checks, 'local_backtracks': backtracks}
        statistics.update(inserted=index, path_checks=checks, backtracks=backtracks)
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
    statistics.update(inserted=index, path_checks=checks, backtracks=backtracks, request_order=[q.id for q in requests], ordinary_assignment_policy='reallocated_by_final_joint_model', draft_capacity_policy='residual_capacity_upper_bound')
    decoder.optimizer.initial_hint = None
    return tuple(plan)


def _construct_preserved_repair(decoder,rule,rng,progress=print,initial=(),pending=None):
    net,p=decoder.net,decoder.p; people=decoder.people
    from .budget import check
    requests=sorted((q for q in net.passengers if q.kind and (pending is None or q.id in pending)),key=lambda q:(q.latest,q.earliest,q.id))
    plan=list(initial);operations={(e.split_trip,e.split_node) for e in plan};occupied={}
    for e in plan: occupied.setdefault(e.split_trip,set()).add(int(e.source.rsplit('m',1)[1]))
    trips=make_trips(net,decoder.timetable,p,operations)
    def evaluate(e,origin):
        result=sequence(net,people,e.events,origin,p)
        if result is None: return None
        distance,wait,node,time=result
        stops=net.lines[origin.line]; position=stops.index(origin.node)
        choices=[]
        for j in range(position+1,len(stops)):
            back=net.distance(node,stops[j])
            tail=sum(net.road(a,b)[0] for a,b in zip(stops[j:],stops[j+1:]))
            choices.append((back+(1-p.coupling_discount)*tail,j,back+tail))
        _,join,extra=min(choices)
        return replace(e,target='',join=join),distance+extra,wait
    for index,q in enumerate(requests):
        check(decoder)
        candidates=[]
        for pos,e in enumerate(plan):
            if not e.events: continue
            trip=trips[e.split_trip]; origin=trip.visits[e.split]
            old=evaluate(e,origin)
            for a in range(len(e.events)+1):
                for b in range(a+1,len(e.events)+2):
                    events=list(e.events);events.insert(a,(q.id,True));events.insert(b,(q.id,False))
                    candidate=evaluate(replace(e,events=tuple(events)),origin)
                    if candidate:
                        end,distance,wait=candidate
                        candidates.append((p.operating_cost*(distance-old[1]),p.time_value*(wait-old[2]),pos,end))
        for trip in trips.values():
            check(decoder)
            used=occupied.get(trip.id,set())
            slot=next((m for m in range(1,p.max_formation) if m not in used),None)
            if slot is None: continue
            for i,v in enumerate(trip.visits[:-1]):
                extra=p.operation_time if (trip.id,v.node) not in operations else 0
                origin=replace(v,departure=v.departure+extra)
                if origin.departure+net.distance(v.node,q.origin)/p.speed*60>q.latest+1e-7: continue
                if extra and any(sequence(net,people,e.events,replace(trip.visits[e.split],departure=trip.visits[e.split].departure+extra),p) is None
                                 for e in plan if e.split_trip==trip.id and e.split>=i): continue
                e=Excursion(f'{trip.id}/m{slot}',i,'',0,((q.id,True),(q.id,False)),trip.id,v.node)
                candidate=evaluate(e,origin)
                if candidate:
                    end,distance,wait=candidate
                    candidates.append((p.purchase_cost+p.operating_cost*distance,p.time_value*wait,None,end))
        if not candidates: raise Infeasible(f'Constructive seed cannot insert DRT {q.id}; {index} requests inserted')
        if rule in (0,1): chosen=min(candidates,key=lambda c:(c[rule],c[1-rule]))
        else:
            nd=[]; best=inf
            for candidate in sorted(candidates,key=lambda c:(c[0],c[1])):
                if candidate[1]<best:
                    nd.append(candidate);best=candidate[1]
            chosen=rng.choice(nd)
        _,_,pos,e=chosen
        if pos is None:
            plan.append(e);occupied.setdefault(e.split_trip,set()).add(int(e.source.rsplit('m',1)[1]))
            operations.add((e.split_trip,e.split_node));trips=make_trips(net,decoder.timetable,p,operations)
        else: plan[pos]=e
        if (index+1)%50==0: progress(f'  seed {rule+1}: routed {index+1}/{len(requests)} DRT in {len(plan)} module chains')
    return tuple(plan)
