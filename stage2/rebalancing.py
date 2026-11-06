from common.model import Excursion
from .insertion import default_returns
from common.roads import RoadPaths

def resource_neighbours(decoder, solution):
    context = decoder.context(solution.excursions)
    net, p = (decoder.net, decoder.p)
    roads = RoadPaths(net, p)
    last = default_returns(context.duties, solution.excursions)
    targets = list(context.trips.values())
    for key, duty in context.duties.items():
        for i, v in enumerate(duty.visits):
            warehouse = v.kind == 'depot'
            if v.node not in net.depots or i < last.get(key, 0) or (not warehouse and (not v.trip or i == len(duty.visits) - 1)):
                continue
            anchor = 'depot' if warehouse else v.trip
            for target in sorted(targets, key=lambda t: (t.line == v.line, t.visits[0].arrival, t.id)):
                if target.id == v.trip:
                    continue
                for j, join in enumerate(target.visits[:-1]):
                    if join.node not in net.depots and not (warehouse and j == 0):
                        continue
                    if target.line == v.line and v.node in net.lines[v.line] and (j <= net.lines[v.line].index(v.node)):
                        continue
                    path = roads.path(v.node, join.node)
                    if path is None:
                        continue
                    prospective = join.departure + (p.operation_time if (target.id, join.node) not in context.operations else 0)
                    if v.departure + path[0] + p.operation_time > prospective + 1e-07:
                        continue
                    release = max(v.departure, join.arrival - path[0]) if warehouse else None
                    yield (solution.excursions + (Excursion(key, i, target.id, j, (), anchor, v.node, release),))
            if not warehouse:
                for depot in net.depots:
                    if depot == v.node:
                        continue
                    if roads.path(v.node, depot) is not None:
                        yield (solution.excursions + (Excursion(key, i, f'depot:{depot}', 0, (), anchor, v.node),))
