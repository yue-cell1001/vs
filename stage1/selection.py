from common.multiobjective import layers2d
from common.pareto import crowding

def select_candidates(candidates, limit=15, p=None, rng=None):
    layers = layers2d([c['objectives'] for c in candidates])
    front = [candidates[i] for i in layers[0]] if layers else []
    kept = list(range(len(front)))
    while len(kept) > limit:
        distance = crowding([front[i]['objectives'] for i in kept])
        index = min(range(len(kept)), key=lambda j: (distance[j], -kept[j]))
        kept.pop(index)
    result = sorted((front[i] for i in kept), key=lambda c: tuple(c['objectives']))
    return (result, {'non_dominated_count': len(front), 'candidate_count': len(result), 'candidate_limit': limit, 'evaluated_feasible_count': len(candidates), 'selection': 'iterative_crowding'})
