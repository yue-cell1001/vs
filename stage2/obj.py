from collections import defaultdict

def evaluate(solution, net, p):
    fixed = defaultdict(list)
    variable = 0.0
    for duty in solution.duties.values():
        for leg in duty.legs:
            if leg.kind == 'fixed':
                fixed[leg.trip, leg.start, leg.end, round(leg.departure, 7)].append(leg)
            else:
                variable += leg.distance * p.operating_cost
    fixed_cost = 0.0
    for group in fixed.values():
        n = len(group)
        fixed_cost += group[0].distance * p.operating_cost * (1 + p.following_ratio * (n - 1))
    rebalance = sum((m['distance'] for m in solution.metadata['rebalance_moves'])) * p.operating_cost
    purchase = solution.metadata['fleet_size'] * p.purchase_cost
    people = {x.id: x for x in net.passengers}
    normal_wait = drt_wait = detour = 0.0
    for pid, s in solution.services.items():
        person = people[pid]
        wait = max(0, s.pickup - person.earliest)
        if person.kind:
            drt_wait += wait
        else:
            normal_wait += wait
            detour += s.offroute_minutes
    solo = sum(leg.distance * p.operating_cost for duty in solution.duties.values() for leg in duty.legs if leg.kind == 'solo')
    solution.components = {'purchase': purchase, 'fixed_travel': fixed_cost, 'drt_travel': variable - solo, 'solo_travel': solo, 'rebalance_travel': rebalance, 'normal_wait_minutes': normal_wait, 'drt_wait_minutes': drt_wait, 'normal_detour_minutes': sum((s.detour for pid, s in solution.services.items() if not people[pid].kind)), 'normal_offroute_minutes': detour, 'drt_detour_minutes': sum(s.detour for pid, s in solution.services.items() if people[pid].kind)}
    solution.objectives = (purchase + fixed_cost + variable + rebalance, normal_wait + drt_wait + sum((s.detour for pid, s in solution.services.items() if not people[pid].kind)))
    return solution.objectives
