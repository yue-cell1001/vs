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
        if e.target not in decoder.nominal_trips:
            return False
        pickups = {q for q, pickup in e.events if pickup}
        drops = {q for q, pickup in e.events if not pickup}
        if pickups != drops:
            return False
        used.add(e.source)
    return True

def repair(decoder, partial, pending, rule, rng):
    required = {q.id for q in decoder.net.passengers if q.kind}
    errors = []
    for strategy_rule in (rule,):
        try:
            decisions = construct(decoder, strategy_rule, rng, lambda _: None, initial=partial, pending=set(pending))
            decoder.preference = 0
            s = decoder.decode(decisions, required, audit=True)
            if not s.errors:
                s.metadata['constructive_repair_rule'] = strategy_rule
                return (s, None)
            errors.append({'rule': strategy_rule, 'errors': s.errors})
        except Infeasible as error:
            errors.append({'rule': strategy_rule, 'errors': [str(error)]})
    return (None, {'reason': 'constructive_repair_failed', 'attempts': errors})
