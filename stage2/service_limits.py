
def direct_drt_time(net, person, p):
    return net.distance(person.origin, person.dest) / p.speed * 60


def completion_limits(trips):
    last = {}
    for trip in trips.values():
        if not trip.id.startswith(trip.line + ':'):
            continue
        if trip.line not in last or trip.visits[0].arrival > last[trip.line].visits[0].arrival:
            last[trip.line] = trip
    return {line: trip.visits[-1].arrival for line, trip in last.items()}


def line_deadline(line, limits):
    return limits.get(line, max(limits.values(), default=float('inf')))


def duty_in_horizon(duty, limits):
    return all(leg.arrival <= line_deadline(leg.line, limits) + 1e-7 for leg in duty.legs)
