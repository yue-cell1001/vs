from dataclasses import replace
from heapq import nsmallest
from common.model import Excursion
from common.pareto import fronts,crowding
from .routing import Infeasible,split_index


def limit(items,count,key=lambda v:v):
    return sorted(items,key=key) if count is None else nsmallest(count,items,key=key)


def prefix(decoder,earlier,e):
    return decoder.context(earlier).duties[e.source]


def event_end(decoder,e,earlier=()):
    p,net=decoder.p,decoder.net
    duty=prefix(decoder,earlier,e); v=duty.visits[split_index(duty,e)]
    node=v.node; time=v.departure; onboard=set(); distance=waits=0.0
    for pid,pickup in e.events:
        person=decoder.people[pid]; dest=person.origin if pickup else person.dest
        d=net.distance(node,dest); distance+=d; time+=d/p.speed*60
        if pickup:
            time=max(time,person.earliest)
            if time>person.latest+1e-7: raise Infeasible('Pickup window')
            waits+=time-person.earliest; onboard.add(pid)
        else: onboard.discard(pid)
        if len(onboard)>p.capacity: raise Infeasible('Instantaneous DRT capacity')
        time+=p.stop_time; node=dest
    return time,node,distance,waits


def endings(decoder,e,allow_depot=True,earlier=()):
    p,net=decoder.p,decoder.net
    try: time,node,_,_=event_end(decoder,e,earlier)
    except Infeasible: return
    context=decoder.context(earlier); source=context.duties[e.source]
    origin=source.visits[split_index(source,e)]; line=origin.line
    position=net.lines[line].index(origin.node)
    choices=[]
    for trip in context.targets.values():
        if trip.id==origin.trip: continue
        for j,v in enumerate(trip.visits[:-1]):
            if trip.line==line and net.lines[line].index(v.node)<=position: continue
            if trip.line!=line and v.node not in net.depots: continue
            travel=net.distance(node,v.node)/p.speed*60


            prospective=v.departure+(p.operation_time if (trip.id,v.node) not in context.operations else 0)
            gap=prospective-time-travel-p.operation_time
            if gap<-1e-7: continue
            if v.arrival-time-travel>p.max_coupling_wait+1e-7 and v.node not in net.depots: continue
            choices.append((net.distance(node,v.node)+max(0,gap)*.05,replace(e,target=trip.id,join=j)))
    for _,end in limit(choices,p.join_limit,key=lambda v:v[0]): yield end

    for j in range(position+1,len(net.lines[line])): yield replace(e,target='',join=j)
    if origin.node in net.depots:
        for other,stops in net.lines.items():
            if other==line: continue
            first=stops.index(origin.node) if origin.node in stops else -1
            for j in range(first+1,len(stops)): yield replace(e,target=f'backup:{other}',join=j)
    if allow_depot:
        for depot in net.depots: yield replace(e,target=f'depot:{depot}',join=0)


def source_options(decoder,person,used=(),limit_count=None,excursions=()):
    context=decoder.context(excursions); choices=[]; signatures=set()


    last=default_returns(context.duties,excursions)
    for key,duty in context.duties.items():
        for i,v in enumerate(duty.visits[:-1]):
            if not v.trip or v.kind in ('pickup','dropoff','empty') or i<last.get(key,0): continue
            travel=decoder.net.distance(v.node,person.origin)/decoder.p.speed*60
            departure=v.departure+(decoder.p.operation_time if (v.trip,v.node) not in context.operations else 0)
            pickup=max(person.earliest,departure+travel)
            if pickup>person.latest+1e-7: continue

            signature=(tuple(duty.visits),tuple(duty.legs),i)
            if signature in signatures: continue
            signatures.add(signature)
            score=travel+.25*(pickup-person.earliest)+.2*max(0,person.earliest-departure-travel)
            choices.append((score,key,i))
    count=decoder.p.source_limit if limit_count is None else limit_count
    return [(k,i) for _,k,i in limit(choices,count,key=lambda v:v[0])]


def default_returns(duties,excursions):
    result={}
    counts={}
    for e in excursions:
        duty=duties[e.source]
        if e.target.startswith('depot:'):
            node=int(e.target.split(':',1)[1])
            candidates=[i for i,v in enumerate(duty.visits) if v.kind=='depot' and v.node==node]
        elif not e.target or e.target.startswith('backup:'):
            counts[e.source]=counts.get(e.source,0)+1
            target=f'backup/{e.source}/{counts[e.source]}'
            candidates=[i for i,v in enumerate(duty.visits) if v.trip==target]
        else:
            candidates=[i for i,v in enumerate(duty.visits) if v.trip==e.target and v.kind in ('join','stored_join')]
        if candidates: result[e.source]=min(candidates) if not e.target or e.target.startswith('backup:') else max(candidates)
    return result


def proposals(decoder,excursions,pid,cooperative=False):
    p=decoder.p; person=decoder.people[pid]

    for pos,e in enumerate(excursions):
        if e.split_trip=='depot': continue
        if p.pool_limit is not None and len({q for q,_ in e.events})>=p.pool_limit: continue
        for i in range(len(e.events)+1):
            for j in range(i+1,len(e.events)+2):
                events=list(e.events); events.insert(i,(pid,True)); events.insert(j,(pid,False))
                tentative=replace(e,events=tuple(events))
                for end in endings(decoder,tentative,earlier=excursions[:pos]):
                    yield tuple(excursions[:pos])+(end,)+tuple(excursions[pos+1:])
    context=decoder.context(excursions)
    sources=source_options(decoder,person,excursions=excursions)
    for key,i in sources:
        v=context.duties[key].visits[i]
        e=Excursion(key,i,'',0,((pid,True),(pid,False)),v.trip,v.node)
        for end in endings(decoder,e,earlier=excursions): yield tuple(excursions)+(end,)
    if cooperative:
        yield from transfer_options(decoder,excursions,person,sources)


def estimate(decoder,candidate,pid):


    affected=[e for e in candidate if any(q==pid for q,_ in e.events)]
    return sum(len(e.events) for e in affected)


def options(decoder,excursions,pid,cooperative=False):
    generated=proposals(decoder,tuple(excursions),pid,cooperative)
    if decoder.p.insertion_limit is None: return generated
    return iter(nsmallest(decoder.p.insertion_limit,generated,key=lambda c:estimate(decoder,c,pid)))


def transfer_options(decoder,excursions,person,sources):

    pid=person.id
    def pickups():
        context=decoder.context(excursions)
        for key,i in sources:
            v=context.duties[key].visits[i]
            pickup=Excursion(key,i,'',0,((pid,True),),v.trip,v.node)
            for end in endings(decoder,pickup,allow_depot=False,earlier=excursions):
                yield tuple(excursions)+(end,),key
        for pos,e in enumerate(excursions):
            if e.split_trip=='depot': continue
            for i in range(len(e.events)+1):
                events=e.events[:i]+((pid,True),)+e.events[i:]
                for end in endings(decoder,replace(e,events=events),allow_depot=False,earlier=excursions[:pos]):
                    yield tuple(excursions[:pos])+(end,)+tuple(excursions[pos+1:]),e.source
    for picked,source in pickups():
        try: context=decoder.context(picked)
        except Infeasible: continue
        first=context.duties[source]
        board=next(i for i,v in enumerate(first.visits) if v.kind=='pickup' and v.node==person.origin)
        trunk={(leg.trip,leg.start,leg.end,round(leg.departure,7))
               for leg in first.legs[board:] if leg.kind=='fixed'}
        def shared_before(duty,index):
            return any((leg.trip,leg.start,leg.end,round(leg.departure,7)) in trunk
                       for leg in duty.legs[:index] if leg.kind=='fixed')

        for pos,e in enumerate(picked):
            if e.source==source or e.split_trip=='depot': continue
            second=context.duties[e.source]
            if not shared_before(second,split_index(second,e)): continue
            for i in range(len(e.events)+1):
                events=e.events[:i]+((pid,False),)+e.events[i:]
                for end in endings(decoder,replace(e,events=events),earlier=picked[:pos]):
                    yield picked[:pos]+(end,)+picked[pos+1:]

        returns=default_returns(context.duties,picked); equivalent=set()
        for key,second in context.duties.items():
            if key==source: continue
            for j,v in enumerate(second.visits[:-1]):
                if not v.trip or j<returns.get(key,0) or not shared_before(second,j): continue
                signature=(tuple(second.visits),tuple(second.legs),j)
                if signature in equivalent: continue
                equivalent.add(signature)
                delivery=Excursion(key,j,'',0,((pid,False),),v.trip,v.node)
                for end in endings(decoder,delivery,earlier=picked): yield picked+(end,)


def original_line_rejoins(decoder,candidate,pid):
    for pos,e in enumerate(candidate):
        if not any(q==pid for q,_ in e.events): continue
        duty=decoder.context(candidate[:pos]).duties[e.source]
        line=duty.visits[split_index(duty,e)].line
        if not e.target: continue
        if e.target.startswith('backup:'):
            if e.target.split(':',1)[1]!=line: return False
        elif e.target.startswith('depot:'): return False
        else:
            context=decoder.context(candidate[:pos])
            if e.target not in context.targets or context.targets[e.target].line!=line: return False
    return True


def insert(decoder, excursions, pid, rule, rng):
    required = {q for e in excursions for q, _ in e.events} | {pid}
    decoder.preference = 0
    choices = []
    seen = set()
    preferred = []
    for candidate in options(decoder, excursions, pid, True):
        if candidate in seen:
            continue
        seen.add(candidate)
        for preference in (0,):
            decoder.preference = preference
            solution = decoder.decode(candidate, required)
            if solution.errors:
                continue
            if rule == 3 and original_line_rejoins(decoder, candidate, pid):
                preferred.append(solution)
            from common.pareto import dominates
            if any((dominates(s.objectives, solution.objectives) for s in choices)):
                continue
            choices = [s for s in choices if not dominates(solution.objectives, s.objectives)] + [solution]
            if rule == 0:
                return solution
    if not choices and rule != 3:
        return insert(decoder, excursions, pid, 3, rng)
    if not choices:
        return None
    if rule == 3 and preferred:
        indices = fronts([s.objectives for s in preferred])[0]
        choices = [preferred[i] for i in indices]
    cd = crowding([s.objectives for s in choices])
    winner = max(range(len(choices)), key=lambda i: (cd[i], rng.random()))
    return choices[winner]
