import math
from common.model import Excursion


def fixed_cuts(net,table,p):
    rows=[]
    for line,stops in net.lines.items():
        people=[q for q in net.passengers if not q.kind and q.line==line]
        for j,(a,b) in enumerate(zip(stops,stops[1:])):
            ids=[q.id for q in people if stops.index(q.origin)<=j<stops.index(q.dest)]
            capacity=p.capacity*p.max_formation*len(table[line])
            if len(ids)>capacity:
                rows.append(dict(line=line,start=a,end=b,passengers=len(ids),scheduled_trips=len(table[line]),
                                 scheduled_capacity=capacity,deficit=len(ids)-capacity,passenger_ids=ids))
    return rows


def relief(decoder,plan,cuts,strategy="capacity_aware"):

    if strategy not in ("capacity_aware","late_departure","legacy_late_departure"):
        raise ValueError("Unknown relief strategy")
    context=decoder.context(plan);result=list(plan)
    used={e.source for e in plan}

    from collections import Counter
    remaining=Counter();demand=Counter()
    for duty in context.duties.values():
        for leg in duty.legs:
            if leg.kind=='fixed':remaining[leg.line,leg.start,leg.end]+=decoder.p.capacity
    for name,stops in decoder.net.lines.items():
        for q in decoder.net.passengers:
            if q.kind or q.line!=name:continue
            for a,b in zip(stops[stops.index(q.origin):stops.index(q.dest)],stops[stops.index(q.origin)+1:stops.index(q.dest)+1]):demand[name,a,b]+=1
    for line in sorted({c['line'] for c in cuts}):
        deficit=max(c['deficit'] for c in cuts if c['line']==line)
        count=math.ceil(deficit/decoder.p.capacity)+decoder.p.initial_relief_reserve
        options=[]
        for key,duty in context.duties.items():
            if key in used or not key.endswith(tuple(f'/m{m}' for m in range(1,decoder.p.max_formation))):continue
            for i,v in enumerate(duty.visits[:-1]):
                if strategy=="legacy_late_departure" and any(e.split_trip==v.trip for e in plan):continue
                if v.line==line or v.node not in decoder.net.depots or v.node in decoder.net.lines[line]:continue

                if (v.trip,v.node) not in context.operations and any(e.split_trip==v.trip and e.split>=i for e in plan):continue
                path=decoder.optimizer.roads.path(v.node,decoder.net.lines[line][0])
                if path is not None:options.append((v.departure,key,i,v))

        for _,key,i,v in sorted(options,key=lambda x:(-x[0],x[1],x[2])):
            if key in used:continue
            removed=Counter((l.line,l.start,l.end) for l in context.duties[key].legs[i:] if l.kind=='fixed')
            if strategy=="capacity_aware" and any(remaining[arc]-decoder.p.capacity*number<min(remaining[arc],demand[arc]) for arc,number in removed.items()):continue
            for arc,number in removed.items():remaining[arc]-=decoder.p.capacity*number
            for a,b in zip(decoder.net.lines[line],decoder.net.lines[line][1:]):remaining[line,a,b]+=decoder.p.capacity
            result.append(Excursion(key,i,'backup:'+line,0,(),v.trip,v.node));used.add(key);count-=1
            if count==0:break
    return tuple(result)
