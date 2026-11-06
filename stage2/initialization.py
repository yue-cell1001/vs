from dataclasses import replace
from math import inf
from common.model import Excursion
from common.timetable import make_trips
from .routing import Infeasible


def sequence(net,people,events,origin,p):
    node=origin.node; time=origin.departure; distance=waiting=0.0; onboard=set()
    for pid,pickup in events:
        q=people[pid]; target=q.origin if pickup else q.dest
        d=net.distance(node,target); distance+=d; time+=d/p.speed*60
        if pickup:
            time=max(time,q.earliest)
            if time>q.latest+1e-7: return None
            onboard.add(pid); waiting+=time-q.earliest
        else:
            if pid not in onboard: return None
            onboard.remove(pid)
        if len(onboard)>p.capacity: return None
        node=target; time+=p.stop_time
    if onboard: return None
    return distance,waiting,node,time


def construct(decoder,rule,rng,progress=print,initial=(),pending=None):
    net,p=decoder.net,decoder.p; people=decoder.people
    from .budget import check
    requests=sorted((q for q in net.passengers if q.kind and (pending is None or q.id in pending)),key=lambda q:(q.latest,q.earliest,q.id))
    plan=list(initial);operations={(e.split_trip,e.split_node) for e in plan};occupied={}
    for e in plan: occupied.setdefault(e.split_trip,set()).add(int(e.source.rsplit('m',1)[1]))
    trips=make_trips(net,decoder.timetable,p,operations)
    def evaluate(e,origin):
        result=sequence(net,people,e.events,origin,p)
        if result is None: return None
        distance,wait,node,time=result
        stops=net.lines[origin.line]; position=stops.index(origin.node)
        choices=[]
        for j in range(position+1,len(stops)):
            back=net.distance(node,stops[j])
            tail=sum(net.road(a,b)[0] for a,b in zip(stops[j:],stops[j+1:]))
            choices.append((back+(1-p.coupling_discount)*tail,j,back+tail))
        _,join,extra=min(choices)
        return replace(e,target='',join=join),distance+extra,wait
    for index,q in enumerate(requests):
        check(decoder)
        candidates=[]
        for pos,e in enumerate(plan):
            if not e.events: continue
            trip=trips[e.split_trip]; origin=trip.visits[e.split]
            old=evaluate(e,origin)
            for a in range(len(e.events)+1):
                for b in range(a+1,len(e.events)+2):
                    events=list(e.events);events.insert(a,(q.id,True));events.insert(b,(q.id,False))
                    candidate=evaluate(replace(e,events=tuple(events)),origin)
                    if candidate:
                        end,distance,wait=candidate
                        candidates.append((p.operating_cost*(distance-old[1]),p.time_value*(wait-old[2]),pos,end))
        for trip in trips.values():
            check(decoder)
            used=occupied.get(trip.id,set())
            slot=next((m for m in range(1,p.max_formation) if m not in used),None)
            if slot is None: continue
            for i,v in enumerate(trip.visits[:-1]):
                extra=p.operation_time if (trip.id,v.node) not in operations else 0
                origin=replace(v,departure=v.departure+extra)
                if origin.departure+net.distance(v.node,q.origin)/p.speed*60>q.latest+1e-7: continue
                if extra and any(sequence(net,people,e.events,replace(trip.visits[e.split],departure=trip.visits[e.split].departure+extra),p) is None
                                 for e in plan if e.split_trip==trip.id and e.split>=i): continue
                e=Excursion(f'{trip.id}/m{slot}',i,'',0,((q.id,True),(q.id,False)),trip.id,v.node)
                candidate=evaluate(e,origin)
                if candidate:
                    end,distance,wait=candidate
                    candidates.append((p.purchase_cost+p.operating_cost*distance,p.time_value*wait,None,end))
        if not candidates: raise Infeasible(f'Constructive seed cannot insert DRT {q.id}; {index} requests inserted')
        if rule in (0,1): chosen=min(candidates,key=lambda c:(c[rule],c[1-rule]))
        else:
            nd=[]; best=inf
            for candidate in sorted(candidates,key=lambda c:(c[0],c[1])):
                if candidate[1]<best:
                    nd.append(candidate);best=candidate[1]
            chosen=rng.choice(nd)
        _,_,pos,e=chosen
        if pos is None:
            plan.append(e);occupied.setdefault(e.split_trip,set()).add(int(e.source.rsplit('m',1)[1]))
            operations.add((e.split_trip,e.split_node));trips=make_trips(net,decoder.timetable,p,operations)
        else: plan[pos]=e
        if (index+1)%50==0: progress(f'  seed {rule+1}: routed {index+1}/{len(requests)} DRT in {len(plan)} module chains')
    return tuple(plan)
