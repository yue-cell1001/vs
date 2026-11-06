import csv
import json
from collections import Counter, defaultdict
from .constraints import validate
from common.timetable import make_trips

def clock(minutes):
    seconds = round(minutes * 60)
    return f'{seconds // 3600:02}:{seconds // 60 % 60:02}:{seconds % 60:02}'

def write_tables(s, net, p, directory, prefix):
    errors = validate(s, net, make_trips(net, s.metadata['timetable'], p), p)
    if errors:
        raise ValueError('Refusing to export incomplete/invalid schedule: ' + '; '.join(errors[:5]))

    def csvfile(suffix, fields, rows):
        with (directory / f'{prefix}_{suffix}.csv').open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    onboard = defaultdict(list)
    for pid, service in s.services.items():
        for r in service.rides:
            for i in range(r.board, r.alight):
                onboard[r.duty, i].append(pid)
    formations = Counter(((leg.trip, leg.start, leg.end, round(leg.departure, 7)) for d in s.duties.values() for leg in d.legs if leg.kind == 'fixed'))
    legs = []
    visits = []
    timetable = {}
    for key, d in s.duties.items():
        for i, v in enumerate(d.visits):
            visits.append(dict(vehicle=d.vehicle, duty=key, visit=i, node=v.node, node_name=net.nodes[v.node].name, arrival=v.arrival, arrival_clock=clock(v.arrival), departure=v.departure, departure_clock=clock(v.departure), kind=v.kind, trip=v.trip, line=v.line))
            if v.trip:
                timetable[v.trip, v.node, v.arrival, v.departure] = (v.trip, v.line, v.node, v.arrival, v.departure)
        for i, l in enumerate(d.legs):
            ids = sorted(onboard[key, i])
            formation = formations[l.trip, l.start, l.end, round(l.departure, 7)] if l.kind == 'fixed' else 1
            legs.append(dict(vehicle=d.vehicle, duty=key, leg=i, start=l.start, end=l.end, departure=l.departure, departure_clock=clock(l.departure), arrival=l.arrival, arrival_clock=clock(l.arrival), km=l.distance, kind=l.kind, trip=l.trip, line=l.line, formation=formation, load=len(ids), passenger_ids=';'.join(map(str, ids))))
    for l in s.metadata['rebalance_moves']:
        legs.append(dict(vehicle=l['vehicle'], duty='', leg='', start=l['start'], end=l['end'], departure=l['departure'], departure_clock=clock(l['departure']), arrival=l['arrival'], arrival_clock=clock(l['arrival']), km=l['distance'], kind='empty_rebalance', trip='', line='', formation=1, load=0, passenger_ids=''))
    legs.sort(key=lambda r: (r['vehicle'], r['departure'], r['arrival']))
    csvfile('module_legs', list(legs[0]) if legs else ['vehicle', 'duty'], legs)
    csvfile('module_visits', list(visits[0]) if visits else ['vehicle', 'duty'], sorted(visits, key=lambda r: (r['vehicle'], r['arrival'], r['departure'])))
    people = {q.id: q for q in net.passengers}
    passengers = []
    rides = []
    for pid, service in sorted(s.services.items()):
        q = people[pid]
        passengers.append(dict(passenger=pid, kind='DRT' if q.kind else 'ordinary', origin=q.origin, destination=q.dest, requested_line=q.line, earliest=q.earliest, latest=q.latest, pickup=service.pickup, pickup_clock=clock(service.pickup), dropoff=service.dropoff, dropoff_clock=clock(service.dropoff), wait=service.pickup - q.earliest, extra_ride_minutes=service.detour, offroute_minutes_eq4=service.offroute_minutes, transfers=len(service.rides) - 1))
        for i, r in enumerate(service.rides):
            d = s.duties[r.duty]
            a, b = (d.visits[r.board], d.visits[r.alight])
            rides.append(dict(passenger=pid, ride=i, vehicle=d.vehicle, duty=r.duty, board_visit=r.board, alight_visit=r.alight, origin=a.node, destination=b.node, board_time=service.pickup if i == 0 else a.departure, alight_time=b.arrival))
    csvfile('passengers', list(passengers[0]) if passengers else ['passenger'], passengers)
    csvfile('passenger_rides', list(rides[0]) if rides else ['passenger'], rides)
    rows = [dict(trip=t, line=l, node=n, arrival=a, arrival_clock=clock(a), departure=b, departure_clock=clock(b)) for t, l, n, a, b in sorted(timetable.values())]
    csvfile('timetable', list(rows[0]) if rows else ['trip'], rows)
    actions = []
    for i, a in enumerate(s.metadata.get('scheduling_actions', ())):
        e = s.excursions[i]
        actions.append(dict(operation=i, vehicle=s.duties[e.source].vehicle, **a, events=json.dumps(e.events, ensure_ascii=False)))
    fields = list(actions[0]) if actions else ['operation', 'vehicle', 'source', 'target']
    csvfile('operations', fields, actions)
    chains = []
    for vehicle, chain in s.metadata['vehicle_chains'].items():
        first = s.duties[chain[0]].visits[0]
        last = s.duties[chain[-1]].visits[-1]
        chains.append(dict(vehicle=vehicle, start_node=first.node, start_time=first.departure, final_node=last.node, finish_time=last.departure, duties=';'.join(chain)))
    csvfile('vehicle_chains', ['vehicle', 'start_node', 'start_time', 'final_node', 'finish_time', 'duties'], chains)
    csvfile('activations', ['vehicle', 'node', 'time', 'origin'], s.metadata['activations'])
    csvfile('storage', ['exchange', 'depot', 'vehicle', 'in', 'out', 'kind'], [dict(exchange=e['node'], depot=net.depots[e['node']], **{k: v for k, v in e.items() if k != 'node'}) for e in s.metadata['storage']])
    inventory = []
    inventory_events = []
    for node, state in s.metadata['inventory'].items():
        inventory_events.extend((dict(exchange=node, depot=net.depots[int(node)], **event) for event in state['events']))
        inventory.append(dict(exchange=node, depot=net.depots[int(node)], initial=state['initial'], final=state['final'], peak=state['peak'], events=json.dumps(state['events'])))
    csvfile('inventory_events', ['exchange', 'depot', 'time', 'change', 'inventory', 'vehicle'], inventory_events)
    csvfile('inventory', ['exchange', 'depot', 'initial', 'final', 'peak', 'events'], inventory)
    audit = {'passed': True, 'errors': errors, 'passengers_expected': len(net.passengers), 'passengers_served': len(s.services), 'DRT_served': sum((people[pid].kind != 0 for pid in s.services)), 'fleet': s.metadata['fleet_size'], 'module_legs': len(legs), 'passenger_rides': len(rides), 'objectives': s.objectives, 'components': s.components}
    (directory / f'{prefix}_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    return audit
