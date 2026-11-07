from .initialization import construct
from .routing import Infeasible

def compatible(decoder, plan):
    used = set()
    for e in plan:
        if e.source in used or e.split_trip not in decoder.nominal_trips:
            return False
        if e.source != e.split_trip + '/' + e.source.rsplit('/', 1)[-1]:
            return False
        if e.source.rsplit('/m', 1)[-1] not in {str(i) for i in range(1, decoder.p.max_formation)}:
            return False
        trip = decoder.nominal_trips[e.split_trip]
        if not 0 <= e.split < len(trip.visits) - 1 or e.split_node != trip.visits[e.split].node:
            return False
        if e.target not in decoder.nominal_trips and e.target not in ('', f'backup:{trip.line}'):
            return False
        if (not e.target or e.target.startswith('backup:')) and not e.split < e.join < len(trip.visits):
            return False
        pickups = {q for q, pickup in e.events if pickup}
        drops = {q for q, pickup in e.events if not pickup}
        if pickups != drops:
            return False
        used.add(e.source)
    return True

def repair(decoder, partial, pending, rule, rng):
    required = {q.id for q in decoder.net.passengers if q.kind}
    try:
        decisions = construct(decoder, rule, rng, lambda _: None, initial=partial, pending=list(pending))
        decoder.preference = 0
        result = decoder.decode(decisions, required, audit=True)
        if not result.errors:
            result.metadata['constructive_repair_rule'] = rule
            return result, None
        errors = result.errors
    except Infeasible as error:
        errors = [str(error)]
    return None, {'reason': 'constructive_repair_failed', 'attempts': [{'rule': rule, 'errors': errors}]}
