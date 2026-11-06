import random
from copy import deepcopy
from math import isfinite
from common.pareto import fronts, crowding, select
from .constraints import repair, count_bounds, violations
from .obj import Evaluator
from .demand import profiles, guided
from .selection import select_candidates
from common.multiobjective import layers2d, hypervolume

def solve(net, p, stats=None, verify_baseline=False):
    rng = random.Random(p.seed)
    low, high = count_bounds(p)
    evaluator = Evaluator(net, p)
    reference = profiles(net, p)
    cache = {}
    bank = {}
    calls = 0
    completed = 0
    stopped = False
    history = []
    frontier = []
    pending_values = []

    def record(iteration):
        nonlocal frontier
        values = frontier + pending_values
        layers = layers2d(values)
        frontier = [values[i] for i in layers[0]] if layers else []
        pending_values.clear()
        history.append({'iteration': iteration, 'objectives': list(frontier), 'evaluation_calls': calls})

    class EvaluationBudgetReached(Exception):
        pass

    def score(table, count=True):
        nonlocal calls
        if count and p.evaluation_budget is not None and calls >= p.evaluation_budget:
            raise EvaluationBudgetReached()
        if count:
            calls += 1
        key = tuple(((line, tuple(t)) for line, t in sorted(table.items())))
        if key not in cache:
            value = evaluator(table) if not violations(table, net, p) else (float('inf'),) * 2
            cache[key] = value
            if all((isfinite(v) for v in value)):
                pending_values.append(value)
                bank[key] = {'timetable': deepcopy(table), 'objectives': value}
        return cache[key]

    def create():
        table = {}
        for line in net.lines:
            times = [p.start]
            while times[-1] < p.end:
                following = times[-1] + rng.uniform(p.min_headway, p.max_headway)
                if following >= p.end:
                    break
                times.append(following)
            table[line] = repair(times + [p.end], p)
        return table
    try:
        number = max(1, min(p.population - 1, round(p.population * p.guided_fraction)))
        population = []
        for attempt in range(p.population * p.repair_budget):
            table = guided(net, p, rng, reference) if len(population) < number else create()
            table = evaluator.decoder.allocation.repair(table)
            if all((isfinite(v) for v in score(table))):
                population.append(table)
            if len(population) == p.population:
                break
        if len(population) < p.population:
            raise ValueError('Initialization budget exhausted; service coverage/boundaries incompatible')

        record(0)

        def mutate(table):
            line = rng.choice(list(table))
            times = table[line]
            kind = rng.choices(['insert', 'delete', 'shift'], weights=[p.insert_probability, p.delete_probability, p.shift_probability])[0]
            proposals = []
            for _ in range(p.mutation_candidates):
                raw = list(times)
                if kind == 'insert' and len(raw) - 1 < high:
                    j = rng.randrange(len(raw) - 1)
                    raw.insert(j + 1, rng.uniform(raw[j], raw[j + 1]))
                elif kind == 'delete' and len(raw) - 1 > low:
                    raw.pop(rng.randrange(1, len(raw) - 1))
                elif kind == 'shift' and len(raw) > 2:
                    j = rng.randrange(1, len(raw) - 1)
                    raw[j] += rng.uniform(-p.mutation_step, p.mutation_step)
                else:
                    continue
                trial = deepcopy(table)
                trial[line] = repair(raw, p)
                proposals.append(trial)
            if not proposals:
                return table
            if rng.random() < p.demand_guided_mutation:
                return min(proposals, key=lambda x: score(x)[1])
            return rng.choice(proposals)
        for generation in range(p.generations):
            values = [score(x, count=False) for x in population]
            rank = {}
            distance = {}
            for level, front in enumerate(fronts(values)):
                for i, d in zip(front, crowding([values[i] for i in front])):
                    rank[i] = level
                    distance[i] = d

            def parent():
                a, b = rng.sample(range(p.population), 2)
                return population[min((a, b), key=lambda i: (rank[i], -distance[i], rng.random()))]
            children = []
            for _ in range(p.population * p.repair_budget):
                a, b = (parent(), parent())
                child = deepcopy(a)
                if rng.random() < p.crossover:
                    for line in child:
                        if rng.random() < 0.5:
                            child[line] = list(b[line])
                        else:
                            cut = rng.uniform(p.start, p.end)
                            child[line] = repair([t for t in a[line] if t <= cut] + [t for t in b[line] if t > cut], p)
                if rng.random() < min(1, p.insert_probability + p.delete_probability + p.shift_probability):
                    child = mutate(child)
                child = evaluator.decoder.allocation.repair(child)
                if all((isfinite(v) for v in score(child))):
                    children.append(child)
                if len(children) == p.population:
                    break
            children.extend(deepcopy(population[:p.population - len(children)]))
            union = population + children
            population = [union[i] for i in select([score(x, count=False) for x in union], p.population)]
            completed = generation + 1
            record(completed)
    except EvaluationBudgetReached:
        stopped = True
        record(completed + 1)
    output, selection = select_candidates(list(bank.values()), limit=p.candidates, p=p, rng=rng)
    checks = []
    if verify_baseline:
        from .verification import verify
        remaining = list(bank.values())
        verified = {}
        while output:
            rejected = set()
            for item in output:
                key = tuple((line, tuple(times)) for line, times in sorted(item['timetable'].items()))
                if key not in verified:
                    report = verify(net, p, item['timetable'], evaluator.decoder.decode(item['timetable']))
                    verified[key] = report
                    checks.append(report)
                    print(f"Ordinary baseline: {report['status']}", flush=True)
                item['ordinary_baseline'] = verified[key]
                if verified[key]['status'] != 'verified_feasible':
                    rejected.add(key)
            if not rejected:
                break
            remaining = [item for item in remaining if tuple((line, tuple(times)) for line, times in sorted(item['timetable'].items())) not in rejected]
            output, selection = select_candidates(remaining, limit=p.candidates, p=p, rng=rng)
    if stats is not None:
        upper = tuple(max(1.0, max((v[i] for v in cache.values() if all(isfinite(x) for x in v)), default=1.0)) for i in range(2))
        stats['ordinary_baseline_checks'] = checks
        stats['convergence'] = [{'iteration': row['iteration'], 'hv': hypervolume(row['objectives'], upper, 0.1), 'evaluation_calls': row['evaluation_calls']} for row in history]
        stats['convergence_scale'] = {'upper': upper, 'margin': 0.1, 'method': 'fixed_post_run_maximum_over_all_evaluated_feasible_points'}
        stats.update(selection)
        stats.update(generations_completed=completed, evaluation_calls=calls, unique_evaluations=len(cache), termination_reason='evaluation_budget' if stopped else 'generation_limit')
    return output
