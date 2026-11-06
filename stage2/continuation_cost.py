def cost(net, p, node, target, join, trips):
    if target in trips:
        trip = trips[target]
        distance = net.distance(node, trip.visits[join].node)
        distance += p.following_ratio * sum(leg.distance for leg in trip.legs[join:])
    elif target.startswith('backup:'):
        route = net.lines[target.split(':', 1)[1]]
        distance = net.distance(node, route[join])
        distance += sum(net.road(a, b)[0] for a, b in zip(route[join:], route[join + 1:]))
    else:
        distance = net.distance(node, int(target.split(':', 1)[1]))
    return p.operating_cost * distance
