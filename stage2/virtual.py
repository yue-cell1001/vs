from common.timetable import Trip

def available_trips(duties, trips):
    result = dict(trips)
    for duty in duties.values():
        identifiers = {v.trip for v in duty.visits if v.trip.startswith(('solo/', 'backup/'))}
        for key in identifiers:
            visits = tuple(v for v in duty.visits if v.trip == key)
            legs = tuple(leg for leg in duty.legs if leg.trip == key and leg.kind == 'solo')
            result[key] = Trip(key, visits[0].line, visits, legs)
    return result
