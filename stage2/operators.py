from math import ceil
NAMES_DESTROY = ('random', 'related', 'marginal', 'chain', 'timing')
NAMES_REPAIR = ('greedy_pareto', 'regret2', 'synchronization')

def remove(excursions,ids,drop_sources=()):

    from dataclasses import replace
    ids=set(ids); dropped=set(); backup_ids={}; counts={}
    for i,e in enumerate(excursions):
        if not e.target or e.target.startswith('backup:'):
            counts[e.source]=counts.get(e.source,0)+1
            backup_ids[i]=f'backup/{e.source}/{counts[e.source]}'
    changed=True
    while changed:
        changed=False; broken_sources=set(drop_sources); broken_backups=set()
        for i,e in enumerate(excursions):
            remaining=[event for event in e.events if event[0] not in ids]
            invalid=(i in dropped or e.source in broken_sources or e.target in broken_backups or
                     bool(e.events and not remaining))
            if not invalid: continue
            broken_sources.add(e.source)
            if i in backup_ids: broken_backups.add(backup_ids[i])
            added={pid for pid,_ in e.events}-ids
            if i not in dropped or added: changed=True
            dropped.add(i); ids.update(added)
    return tuple(replace(e,events=tuple(event for event in e.events if event[0] not in ids))
                 for i,e in enumerate(excursions) if i not in dropped)

def destroy(decoder, solution, operator, rng):
    ids = sorted({pid for e in solution.excursions for pid, _ in e.events})
    if not ids:
        if solution.excursions and operator in (3, 4):
            return (remove(solution.excursions, (), {rng.choice(solution.excursions).source}), [])
        return (solution.excursions, [])
    fraction = decoder.p.destroy_fraction
    n = min(len(ids), max(1, ceil(fraction * len(ids))))
    if operator == 0:
        selected = rng.sample(ids, n)
    elif operator == 1:
        seed = decoder.people[rng.choice(ids)]
        selected = sorted(ids, key=lambda pid: (decoder.net.distance(decoder.people[pid].origin, seed.origin) + decoder.net.distance(decoder.people[pid].dest, seed.dest) + abs(decoder.people[pid].earliest - seed.earliest) / 30, -sum((any((pid == q for q, _ in e.events)) and any((seed.id == q for q, _ in e.events)) for e in solution.excursions))))[:n]
    elif operator == 2:
        upper = getattr(decoder, 'upper', (1.0, 1.0))
        scores = {pid: [0.0, solution.services[pid].pickup - decoder.people[pid].earliest] for pid in ids}
        for pid in ids:
            for e in solution.excursions:
                if any((q == pid for q, _ in e.events)):
                    d = solution.duties[e.source]
                    count = max(1, len({q for q, _ in e.events}))
                    scores[pid][0] += sum((l.distance * decoder.p.operating_cost for l in d.legs if l.kind == 'drt')) / count
                    scores[pid][1] += sum((s.detour for q, s in solution.services.items() if not decoder.people[q].kind and any((r.duty == e.source for r in s.rides)))) / count
        selected = sorted(ids, key=lambda pid: -max(scores[pid][0] / upper[0], scores[pid][1] / upper[1]))[:n]
    elif operator == 3:
        chains = {}
        for e in solution.excursions:
            chains.setdefault(solution.duties[e.source].vehicle, []).append(e)

        def difficulty(item):
            vehicle, events = item
            waits = sum((v.departure - v.arrival for duty in solution.duties.values() if duty.vehicle == vehicle for v in duty.visits if v.kind in ('join', 'stored_join')))
            return waits + 1 / max(1, len(solution.metadata.get('vehicle_chains', {}).get(vehicle, [])))
        _, events = max(chains.items(), key=difficulty)
        selected = list({pid for e in events for pid, _ in e.events})
    else:
        anchor = rng.choice(solution.excursions)
        related = {anchor.target, anchor.split_trip}
        for trip in tuple(related):
            if ':' not in trip:
                continue
            line, index = trip.rsplit(':', 1)
            if index.isdigit():
                related.update((f'{line}:{i}' for i in (int(index) - 1, int(index) + 1) if i >= 0))
        events = [e for e in solution.excursions if e.target in related or e.split_trip in related]
        selected = list({pid for e in events for pid, _ in e.events})
        decoder.released_positions = {(line, i) for trip in related if ':' in trip for line, index in [trip.rsplit(':', 1)] if index.isdigit() for i in [int(index)]}
    partial = remove(solution.excursions, selected)
    return (partial, sorted(set(ids) - {q for e in partial for q, _ in e.events}))
