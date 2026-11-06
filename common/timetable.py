from dataclasses import dataclass
from .model import Visit, Leg

@dataclass(frozen=True)
class Trip:
    id: str
    line: str
    visits: tuple[Visit, ...]
    legs: tuple[Leg, ...]

def make_trips(net, timetable, params, operations=None):
    operations = {tuple(action) for action in operations or ()}
    trips = {}
    conditional = getattr(params, 'conditional_operations', True)
    for line, times in sorted(timetable.items()):
        stops = net.lines[line]
        for f, origin_time in enumerate(times):
            key = f'{line}:{f}'

            def dwell(node):
                return params.stop_time + (params.operation_time if not conditional or (key, node) in operations else 0)
            t = origin_time + dwell(stops[0])
            visits = [Visit(stops[0], origin_time, t, key, line)]
            legs = []
            for a, b in zip(stops, stops[1:]):
                distance, travel = net.road(a, b)
                legs.append(Leg(a, b, t, t + travel, distance, 'fixed', key, line))
                arrival = t + travel
                t = arrival + dwell(b)
                visits.append(Visit(b, arrival, t, key, line))
            trips[key] = Trip(key, line, tuple(visits), tuple(legs))
    return trips
