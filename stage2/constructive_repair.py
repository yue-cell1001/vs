from .initialization import construct
from .routing import Infeasible


def compatible(decoder,plan):
    used=set()
    for e in plan:
        if e.source in used or e.split_trip not in decoder.nominal_trips:return False
        if e.source!=e.split_trip+'/'+e.source.rsplit('/',1)[-1]:return False
        if e.source.rsplit('/m',1)[-1] not in {str(i) for i in range(1,decoder.p.max_formation)}:return False
        trip=decoder.nominal_trips[e.split_trip]
        if not 0<=e.split<len(trip.visits)-1 or e.split_node!=trip.visits[e.split].node:return False
        if e.target and not e.target.startswith("backup:"):return False
        if e.events and e.target:return False
        pickups={q for q,pickup in e.events if pickup};drops={q for q,pickup in e.events if not pickup}
        if pickups!=drops:return False
        used.add(e.source)
    return True


def repair(decoder,partial,pending,rule,rng):
    required={q.id for q in decoder.net.passengers if q.kind}
    errors=[]
    from .recovery import fixed_cuts,relief

    for strategy_rule in dict.fromkeys((rule,0,1,2)):
        try:
            decisions=construct(decoder,strategy_rule,rng,lambda _:None,initial=partial,pending=set(pending))
            decoder.preference=rule if rule in (0,1) else rng.randrange(2)
            s=decoder.decode(decisions,required,audit=True)
            if not s.errors:
                s.metadata['constructive_repair_rule']=strategy_rule
                return s,None
            errors.append({'rule':strategy_rule,'errors':s.errors})
            for strategy in ('capacity_aware','late_departure'):
                supported=relief(decoder,decisions,fixed_cuts(decoder.net,decoder.timetable,decoder.p),strategy)
                if supported==decisions:continue
                candidate=decoder.decode(supported,required,audit=True)
                if not candidate.errors:
                    candidate.metadata['repair_recovery']=strategy
                    candidate.metadata['constructive_repair_rule']=strategy_rule
                    return candidate,None
                errors.append({'rule':strategy_rule,'recovery':strategy,'errors':candidate.errors})
        except Infeasible as error:
            errors.append({'rule':strategy_rule,'errors':[str(error)]})
    return None,{'reason':'constructive_repair_failed','attempts':errors}
