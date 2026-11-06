from bisect import bisect_left, bisect_right
from .budget import check
from .continuation_cost import cost


def endings(decoder, trips, operations, origin, excursion, node, earliest, latest, position, distance):
    net, p = decoder.net, decoder.p
    mask = frozenset(operations)
    cache = getattr(decoder, 'return_index_cache', None)
    if cache is None:
        cache = decoder.return_index_cache = {}
    if mask not in cache:
        rows = {}
        for target in trips.values():
            for j, visit in enumerate(target.visits[:-1]):
                couple = visit.departure + (p.operation_time if (target.id, visit.node) not in mask else 0) - p.operation_time
                rows.setdefault(visit.node, []).append((couple, target.id, j))
        indexed = {}
        for station, values in rows.items():
            values.sort()
            indexed[station] = ([row[0] for row in values], values)
        if len(cache) >= 256:
            cache.pop(next(iter(cache)))
        cache[mask] = indexed
    choices = []
    warehouse = excursion.split_trip == 'depot'
    for station, (times, values) in cache[mask].items():
        check(decoder)
        if not warehouse and station not in net.lines[origin.line] and origin.node not in net.depots:
            continue
        back = distance(node, station)
        travel = back / p.speed * 60
        lo = bisect_left(times, earliest + travel - 1e-7)
        hi = bisect_right(times, latest + travel + p.max_coupling_wait + 1e-7)
        for couple, target_id, j in values[lo:hi]:
            target = trips[target_id]
            if not warehouse:
                if target.line == origin.line:
                    if int(target.id.rsplit(':', 1)[1]) <= int(origin.trip.rsplit(':', 1)[1]) or j <= position:
                        continue
                elif origin.node not in net.depots or station not in net.depots:
                    continue
            choices.append((cost(net, p, node, target_id, j, trips), couple - earliest - travel, target_id, j))
    for depot in net.depots:
        target = f'depot:{depot}'
        choices.append((cost(net, p, node, target, 0, trips), 0.0, target, 0))
    if decoder.allow_solo and origin.line in net.lines:
        route = net.lines[origin.line]
        for j in range(position + 1, len(route) - 1):
            target = f'backup:{origin.line}'
            choices.append((cost(net, p, node, target, j, trips), 0.0, target, j))
    for _, _, target, j in sorted(choices):
        yield target, j
