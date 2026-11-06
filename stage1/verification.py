from dataclasses import fields
from stage2.params import Params
from stage2.decoder import Decoder


def verify(net, p, table, reference):
    parameters = {field.name: getattr(p, field.name) for field in fields(Params) if hasattr(p, field.name)}
    decoder = Decoder(net, table, Params(**parameters))
    people = {q.id: q for q in net.passengers}
    active, rides = set(), {}
    for trip, record in reference.trips.items():
        count = record['formation']
        active.update(f'{trip}/m{i}' for i in range(count))
        seats = [[-1] * p.capacity for _ in range(count)]
        stops = net.lines[record['line']]
        passengers = [(stops.index(people[pid].origin), stops.index(people[pid].dest), pid) for pid, ride in reference.assignments.items() if ride['trip'] == trip]
        for a, b, pid in sorted(passengers):
            allocated = False
            for m, available in enumerate(seats):
                seat = next((i for i, end in enumerate(available) if end <= a), None)
                if seat is not None:
                    available[seat] = b
                    rides[pid] = (f'{trip}/m{m}', a, b)
                    allocated = True
                    break
            if not allocated:
                return {'status': 'unresolved', 'errors': ['Ordinary seat assignment not constructed']}
    decoder.optimizer.initial_hint = {'active': active, 'ordinary_rides': rides}
    decoder.optimizer.verify_only = True
    solution = decoder.decode((), set(), audit=True)
    if solution.errors:
        proven = any('Joint integer allocation: Infeasible;' in error for error in solution.errors)
        return {'status': 'infeasible' if proven else 'unresolved', 'errors': solution.errors}
    return {'status': 'verified_feasible', 'ordinary_passengers': len(solution.services), 'full_drt_feasibility_verified': False}
