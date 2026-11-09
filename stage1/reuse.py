from copy import deepcopy
from .constraints import violations
from .reference import ReferenceDecoder
from .verification import verify


def refresh(items, net, p):
    if not isinstance(items, list) or not 0 < len(items) <= p.candidates:
        raise ValueError('Missing or invalid saved first-stage timetables')
    decoder = ReferenceDecoder(net, p)
    result = []
    for item in items:
        table = item['timetable']
        if violations(table, net, p):
            raise ValueError('Saved first-stage timetable violates current timing constraints')
        reference = decoder.decode(table)
        if reference.uncovered:
            raise ValueError('Saved first-stage timetable cannot cover current ordinary demand')
        report = verify(net, p, table, reference)
        if report['status'] != 'verified_feasible':
            raise ValueError('Saved first-stage timetable failed current ordinary baseline verification')
        result.append({**deepcopy(item), 'objectives': reference.objectives, 'ordinary_baseline': report})
    return result
