from common.timetable import make_trips

def objective_bounds(net, candidates, p):
    if not candidates:
        raise ValueError('No branch timetables')
    latest = p.start
    earliest = p.start
    module_bound = 0
    for item in candidates:
        table = item.get('timetable', item)
        module_bound = max(module_bound, sum((len(t) for t in table.values())) * p.max_formation)
        for trip in make_trips(net, table, p, {(f'{line}:{i}', node) for line, times in table.items() for i in range(len(times)) for node in net.lines[line]}).values():
            latest = max(latest, trip.visits[-1].departure + p.shift_limit)
            earliest = min(earliest, trip.visits[0].arrival - p.shift_limit)
    module_bound += len(net.depots) * p.depot_initial
    final_empty_bound = sum((t for _, t in net.edges.values())) + p.operation_time
    horizon = max(1, latest - earliest + final_empty_bound)
    speed = max([p.speed / 60] + [d / t for d, t in net.edges.values() if t > 0])
    u1 = max(1, module_bound * (p.purchase_cost + p.operating_cost * speed * horizon))
    u2 = max(1, 2 * len(net.passengers) * horizon)
    return ((u1, u2), {'module_bound': module_bound, 'earliest': earliest, 'passenger_completion_bound': latest, 'duration_bound': horizon, 'distance_per_minute_bound': speed})
