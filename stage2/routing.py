from dataclasses import replace
from common.model import Duty, Visit, Leg

class Infeasible(ValueError):
    pass

def split_index(duty, e):
    if e.split_trip == 'depot':
        indices = [i for i, v in enumerate(duty.visits) if v.kind == 'depot' and v.node == e.split_node]
        if not indices:
            raise Infeasible('Warehouse continuation has no arriving module')
        return indices[-1]
    if e.split_trip:
        indices = [i for i, v in enumerate(duty.visits[:-1]) if v.trip == e.split_trip and v.node == e.split_node]
        if not indices:
            raise Infeasible('Split anchor not on current module chain')
        return indices[-1]
    return e.split

def construct_duties(net, trips, base, excursions, p, relaxed=False, with_operations=False, operation_schedule=(), action_log=None):
    duties = {k: Duty(k, list(v.visits), list(v.legs)) for k, v in base.items()}
    people = {x.id: x for x in net.passengers}
    last_return = {}
    operations = set()
    counts = {}
    operation_schedule = set(operation_schedule)
    from common.roads import RoadPaths
    roads = RoadPaths(net, p)
    for e in excursions:
        counts[e.source] = counts.get(e.source, 0) + 1
        if e.source not in duties:
            raise Infeasible('Unknown source module')
        original = duties[e.source]
        index = split_index(original, e)
        if not 0 <= index < len(original.visits) or (index == len(original.visits) - 1 and original.visits[index].kind != 'depot'):
            raise Infeasible('Invalid split')
        if index < last_return.get(e.source, 0):
            raise Infeasible('Overlapping excursions on one module chain')
        origin = original.visits[index]
        warehouse = origin.kind == 'depot'
        if not warehouse and (not origin.trip or origin.node not in net.lines.get(origin.line, ())):
            raise Infeasible('Split must occur at a fixed-line station')
        line = origin.line
        stops = net.lines.get(line, ())
        position = stops.index(origin.node) if origin.node in stops else -1
        if not warehouse:
            operations.add((origin.trip, origin.node))
        visits = list(original.visits[:index + 1])
        legs = list(original.legs[:index])
        current = origin.node
        t = origin.departure
        seen = set()
        if warehouse and e.release_time is not None:
            if e.release_time < origin.departure - 1e-07:
                raise Infeasible('Warehouse release precedes module arrival/readiness')
            t = e.release_time
            visits[-1] = replace(origin, departure=t)
        event_times = None
        if e.events and not relaxed and e.target in trips:
            from .synchronization import service_times
            join = trips[e.target].visits[e.join]
            event_times = service_times(net, people, e.events, origin.node, t, join.node, join.departure - p.operation_time, p)
        for event_index, (pid, pickup) in enumerate(e.events):
            if (pid, pickup) in seen:
                raise Infeasible('Repeated pickup/drop event')
            seen.add((pid, pickup))
            person = people[pid]
            if not person.kind:
                raise Infeasible('Only DRT passengers define service events')
            node = person.origin if pickup else person.dest
            distance = net.distance(current, node)
            arrival = t + distance / p.speed * 60
            desired = event_times[event_index] if event_times is not None else person.earliest
            if pickup and arrival < desired - 1e-07:
                legs.append(Leg(current, node, t, arrival, distance, 'drt', '', line))
                visits.append(Visit(node, arrival, desired, '', line, 'pickup_wait'))
                current = node
                t = desired
                arrival = t
                distance = 0
            if not relaxed and pickup and (arrival > person.latest + 1e-07):
                raise Infeasible('Pickup window')
            legs.append(Leg(current, node, t, arrival, distance, 'drt', '', line))
            t = arrival + p.stop_time
            visits.append(Visit(node, arrival, t, '', line, 'pickup' if pickup else 'dropoff'))
            current = node

        def travel_to(node):
            nonlocal current, t
            if e.events:
                distance = net.distance(current, node)
                arrival = t + distance / p.speed * 60
                legs.append(Leg(current, node, t, arrival, distance, 'drt', '', line))
                t = arrival
                current = node
            else:
                path = roads.path(current, node)
                if path is None:
                    raise Infeasible('No directed empty-dispatch road path')
                for a, b, distance, travel in path[2]:
                    legs.append(Leg(a, b, t, t + travel, distance, 'empty', '', line))
                    t += travel
                    visits.append(Visit(b, t, t, '', line, 'empty'))
                    current = b
                if not path[2]:
                    legs.append(Leg(current, node, t, t, 0, 'empty', '', line))
                elif visits[-1].node == node:
                    visits.pop()
                current = node
        if e.target.startswith('depot:'):
            node = int(e.target.split(':')[1])
            if node not in net.depots:
                raise Infeasible('Not a storage interchange')
            travel_to(node)
            visits.append(Visit(node, t, t + p.operation_time, '', line, 'depot'))
            return_index = len(visits) - 1
        elif not e.target or e.target.startswith('backup:'):
            solo_line = e.target.split(':', 1)[1] if e.target else line
            if solo_line not in net.lines or (line and solo_line != line):
                raise Infeasible('Independent continuation must retain its original direction')
            route = net.lines[solo_line]
            if not 0 <= e.join < len(route) or (not warehouse and e.join <= position):
                raise Infeasible('Independent return must be downstream with a remaining route segment')
            node = route[e.join]
            travel_to(node)
            solo = f'backup/{e.source}/{counts[e.source]}'
            visits.append(Visit(node, t, t + p.stop_time + (p.operation_time if (solo, node) in operation_schedule else 0), solo, solo_line, 'solo'))
            t = visits[-1].departure
            return_index = len(visits) - 1
            for a, b in zip(route[e.join:], route[e.join + 1:]):
                distance, travel = net.road(a, b)
                legs.append(Leg(a, b, t, t + travel, distance, 'solo', solo, solo_line))
                arrival = t + travel
                t = arrival + p.stop_time + (p.operation_time if (solo, b) in operation_schedule else 0)
                visits.append(Visit(b, arrival, t, solo, solo_line, 'solo'))
        elif not e.target:
            raise Infeasible('Missing post-DRT destination')
        else:
            from .virtual import available_trips
            targets = available_trips(duties, trips)
            if e.target not in targets:
                raise Infeasible('Unknown target trip')
            target = targets[e.target]
            if e.target not in trips:
                raise Infeasible('Target must be a scheduled trip')
            if not 0 <= e.join < len(target.visits) - 1:
                raise Infeasible('Invalid join')
            if target.id == origin.trip:
                raise Infeasible('Cannot rejoin the original trip')
            join = target.visits[e.join]
            if target.line == line and net.lines[line].index(join.node) <= position:
                raise Infeasible('Rejoin must be downstream')
            if target.line == line and target.id in trips and (origin.trip in trips) and (int(target.id.rsplit(':', 1)[1]) <= int(origin.trip.rsplit(':', 1)[1])):
                raise Infeasible('Same-line rejoin must use a later scheduled trip')
            if target.line != line and join.node not in net.depots and not warehouse:
                raise Infeasible('Cross-line join outside interchange')
            if target.line != line and (not warehouse) and (origin.node not in net.depots):
                raise Infeasible('Empty cross-line dispatch must leave an interchange')
            operations.add((target.id, join.node))
            travel_to(join.node)
            arrival = t
            ready = arrival + p.operation_time
            if not relaxed and ready > join.departure + 1e-07:
                raise Infeasible('Missed coupling slot')
            wait = max(0, join.departure - p.operation_time - arrival)
            if not relaxed and wait > p.max_coupling_wait + 1e-07 and (e.events or join.node not in net.depots):
                raise Infeasible('Excess waiting outside a depot')
            kind = 'stored_join' if wait > p.max_coupling_wait + 1e-07 else 'join'
            if target.line != line and join.node in stops:
                visits.append(Visit(join.node, arrival, arrival, '', line, 'return'))
                legs.append(Leg(join.node, join.node, arrival, arrival, 0, 'couple', '', line))
            visits.append(replace(join, arrival=arrival, kind=kind))
            return_index = len(visits) - 1
            visits.extend(target.visits[e.join + 1:])
            legs.extend(target.legs[e.join:])
        duties[e.source] = Duty(e.source, visits, legs)
        last_return[e.source] = return_index
        if action_log is not None:
            action_log.append({'source': e.source, 'split_trip': origin.trip, 'split_node': origin.node, 'source_line': line, 'target_line': visits[-1].line, 'target': e.target, 'departure': visits[index].departure, 'return_arrival': visits[return_index].arrival, 'waiting_index': f'res/{e.source}/{counts[e.source]}' if e.events else None, 'coupling_complete': visits[return_index].departure})
    if with_operations:
        return (duties, operations)
    return duties

def cross_line_spans(duties, trips, excursions):
    result = {}
    for e in excursions:
        if e.source not in duties or e.target not in trips:
            continue
        duty = duties[e.source]
        start = split_index(duty, e)
        if duty.visits[start].line == trips[e.target].line:
            continue
        node = trips[e.target].visits[e.join].node
        end = next((i for i, v in enumerate(duty.visits) if i >= start and v.trip == e.target and (v.node == node) and (v.kind in ('join', 'stored_join'))), None)
        if end is None:
            raise Infeasible('Missing cross-line coupling event')
        result.setdefault(e.source, []).append((start, end))
    return result
