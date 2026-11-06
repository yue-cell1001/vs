from .resources import inventory


def check_inventory(duties, excursions, net, p):
    storage, activations = [], []
    for key in {e.source for e in excursions}:
        duty = duties[key]
        first = duty.visits[0]
        if key.startswith('reserve:'):
            activations.append({'vehicle': key, 'node': first.node, 'time': first.departure, 'origin': 'reserve'})
        for visit in duty.visits[1:-1]:
            if visit.kind == 'depot' and visit.departure > visit.arrival + p.operation_time + 1e-7:
                storage.append({'node': visit.node, 'vehicle': key, 'in': visit.arrival + p.operation_time, 'out': visit.departure})
            if visit.kind == 'stored_join':
                storage.append({'node': visit.node, 'vehicle': key, 'in': visit.arrival + p.operation_time, 'out': visit.departure - p.operation_time})
        end = duty.visits[-1]
        if end.kind == 'depot':
            storage.append({'node': end.node, 'vehicle': key, 'in': end.departure, 'out': None})
    return inventory(storage, net, p, activations)
