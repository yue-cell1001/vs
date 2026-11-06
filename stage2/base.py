from common.model import Duty, Visit

def build_base(net, trips, p, sources=None):
    result = {f'{key}/m{m}': Duty(f'{key}/m{m}', list(trip.visits), list(trip.legs)) for key, trip in trips.items() for m in range(p.max_formation) if sources is None or f'{key}/m{m}' in sources}
    for node in net.depots:
        for slot in range(p.depot_capacity):
            key = f'reserve:{node}/m{slot}'
            if sources is not None and key not in sources:
                continue
            result[key] = Duty(key, [Visit(node, p.start, p.start, '', '', 'depot')], [])
    return result
