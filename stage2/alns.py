import random
from collections import defaultdict
from time import monotonic
from common.pareto import Archive, crowding, dominates, objective_key
from common.multiobjective import hypervolume
from .decoder import Decoder
from .constraints import validate
from .initialization import construct
from .routing import Infeasible
from .operators import destroy
from .insertion import insert
from .learning import AdaptiveAutomaton as Automaton
from .timing import shifted_candidates, validate_branch
from .budget import SearchTimeout
from .scalarization import DIRECTIONS, positive_pair

class NoFeasibleSolution(RuntimeError):
    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}

def _tuple(value):
    return tuple((_tuple(x) for x in value)) if isinstance(value, list) else value

def repair(decoder,excursions,pending,rule,rng):
    if decoder.p.repair_mode=='constructive':
        from .constructive_repair import compatible,repair as constructive
        from .routing import Infeasible
        if compatible(decoder,excursions):
            try:return constructive(decoder,excursions,pending,rule,rng)
            except Infeasible as error:return None,{'reason':'constructive_repair_failed','errors':[str(error)]}
    working=tuple(excursions); solution=None
    if not pending:
        decoder.preference=rule if rule in (0,1) else rng.randrange(2)
        solution=decoder.decode(working,{q for e in working for q,_ in e.events})
        if solution.errors: return None,None
    for pid in pending:
        solution=insert(decoder,working,pid,rule,rng)
        if solution is None: return None,pid
        working=solution.excursions
    if rule==2 and solution is not None:
        from .rebalancing import resource_neighbours
        from common.pareto import dominates
        best=solution
        for index,candidate in enumerate(resource_neighbours(decoder,solution)):
            if decoder.p.insertion_limit is not None and index>=decoder.p.insertion_limit: break
            trial=decoder.decode(candidate,{q for e in solution.excursions for q,_ in e.events})
            if not trial.errors and dominates(trial.objectives,best.objectives): best=trial
        solution=best
    return solution,None

def choose_baseline(items, rng):
    distance = crowding([s.objectives for s in items])
    a, b = rng.choices(range(len(items)), k=2)
    return items[a if distance[a] >= distance[b] else b]

def solve(net, timetable, p, progress=print, seeds=(), checkpoint=None, resume=None, global_archive=None, upper=None, allocation_scales=None):
    from .bounds import objective_bounds
    start = monotonic()
    rng = random.Random(p.seed)
    archive = Archive()
    global_archive = global_archive if global_archive is not None else Archive()
    upper = upper or objective_bounds(net, [{'timetable': timetable}], p)[0]
    allocation_scales = positive_pair(allocation_scales or upper)
    actions = [(d, r) for d in range(5) for r in range(3)]
    learner = Automaton(actions, p.learning_rate, p.exploration)
    failures = []
    trace = []
    offset = 0
    previous_elapsed = 0
    initialization_elapsed = 0.0
    initialization_construction_seconds = 0.0
    initialization_decode_seconds = 0.0
    initialization_attempts = []
    stall = 0
    termination = 'iterations'
    if resume:
        archive.items = list(resume['solutions'])
        state = resume['state']
        if tuple(state.get('allocation_scales', ())) != allocation_scales:
            raise ValueError('Resume objective scale mismatch')
        offset = state['iteration']
        trace = state['trace']
        rng.setstate(_tuple(state['rng']))
        learner.restore(state['learner'])
        previous_elapsed = state.get('elapsed_seconds', 0)
        failures = state.get('initial_failures', [])
        initialization_elapsed = state.get('initialization_elapsed_seconds', 0.0)
        initialization_construction_seconds = state.get('initialization_construction_seconds', 0.0)
        initialization_decode_seconds = state.get('initialization_decode_seconds', 0.0)
        initialization_attempts = state.get('initialization_attempts', [])
        stall = state.get('stall', 0)
        for s in archive.items:
            global_archive.add(s)
    else:
        decoder = Decoder(net, timetable, p)
        decoder.objective_scales = allocation_scales
        decoder.deadline = monotonic() + p.initialization_seconds if p.initialization_seconds else None
        def initial_decode(*args, **kwargs):
            nonlocal initialization_decode_seconds
            before = monotonic()
            progress('  Initial joint allocation started')
            try:
                return decoder.decode(*args, **kwargs)
            finally:
                elapsed = monotonic() - before
                initialization_decode_seconds += elapsed
                progress(f'  Initial joint allocation finished in {elapsed:.2f} seconds')

        def initial_construct(rule):
            nonlocal initialization_construction_seconds
            from .initialization import construct
            before = monotonic()
            try:
                return construct(decoder, rule, rng, progress)
            finally:
                elapsed = monotonic() - before
                initialization_construction_seconds += elapsed
                initialization_attempts.append({'rule': rule, 'strategy': 'uploaded', 'construction_seconds': elapsed, 'statistics': dict(getattr(decoder, 'initialization_stats', getattr(decoder, 'last_construction', {})))})
                progress(f'  Initial route construction finished in {elapsed:.2f} seconds')

        requests = sorted((x for x in net.passengers if x.kind), key=lambda x: (x.latest, x.id))
        seeds=tuple(seeds)
        for decisions in seeds:
            for preference in (0,1):
                decoder.preference=preference
                initial=initial_decode(tuple(decisions),{q.id for q in requests},audit=True)
                if initial.errors: failures.append({'warm_start':True,'errors':initial.errors})
                else: archive.add(initial)
        for rule in range(0 if seeds or resume is not None else p.initial_solutions):
            if p.initial_feasible_target is not None and len(archive.items)>=p.initial_feasible_target:break
            decoder.preference=rule%2
            if p.initialization=='constructive':
                from .initialization import construct
                from .routing import Infeasible
                try:
                    decisions=initial_construct(rule)
                    initial=initial_decode(decisions,{q.id for q in requests},audit=True)
                    if initial.errors:
                        failures.append({'initial_rule': rule, 'errors': initial.errors})
                    if not initial.errors: archive.add(initial)
                except (Infeasible, SearchTimeout) as error:
                    failures.append({'initial_rule':rule,'error':str(error), 'timed_out': isinstance(error, SearchTimeout)})
                    if isinstance(error, SearchTimeout):
                        break
            else:
                decoder.preference=rule%2
                excursions=(); initial=None
                for index,person in enumerate(requests):
                    initial=insert(decoder,excursions,person.id,rule%4,rng)
                    if initial is None:
                        failures.append({'initial_rule':rule,'passenger':person.id,'inserted':index})
                        progress(f'  initial {rule+1}: infeasible insertion at DRT {person.id} ({index}/{len(requests)})')
                        break
                    excursions=initial.excursions
                    if (index+1)%50==0: progress(f'  initial {rule+1}: {index+1}/{len(requests)} DRT, {monotonic()-start:.1f}s')
                else:
                    if initial is None: initial=initial_decode((),set(),audit=True)
                    initial.errors=validate(initial,net,decoder.trips,p)
                    archive.add(initial)

        if not archive.items:
            raise NoFeasibleSolution(f'No complete feasible initial schedule; {failures}. No requests were silently dropped. This initialization failure is not a timetable infeasibility proof; try other initial strategies or diagnose fixed-path capacity.', {'initialization_elapsed_seconds': monotonic() - start, 'initialization_construction_seconds': initialization_construction_seconds, 'initialization_decode_seconds': initialization_decode_seconds, 'initialization_attempts': initialization_attempts, 'initial_failures': failures})
        for candidate in archive.items:
            candidate.metadata['branch_base'] = timetable
            global_archive.add(candidate)
        initialization_elapsed = monotonic() - start

    def state(iteration):
        return {'iteration': iteration, 'rng': rng.getstate(), 'learner': learner.snapshot(), 'trace': trace, 'base_timetable': timetable, 'initial_failures': failures, 'objective_bounds': upper, 'allocation_scales': allocation_scales, 'stall': stall, 'initialization_elapsed_seconds': initialization_elapsed, 'initialization_construction_seconds': initialization_construction_seconds, 'initialization_decode_seconds': initialization_decode_seconds, 'initialization_attempts': initialization_attempts, 'elapsed_seconds': previous_elapsed + monotonic() - start}
    search_deadline = monotonic() + p.search_seconds if p.search_seconds else None
    if checkpoint:
        checkpoint(archive.items, {'phase': 'search', 'iteration': offset, 'state': state(offset)})
    for iteration in range(offset, p.iterations):
        if search_deadline is not None and monotonic() >= search_deadline:
            termination = 'time_budget'
            break
        if stall >= p.stall:
            termination = 'stall'
            break
        current = choose_baseline(archive.items, rng)
        available = [a for a in actions if a[0] == 4 and p.shift_limit > 0 and any((len(t) > 2 for t in timetable.values())) or (a[0] != 4 and bool(current.excursions))]
        if not available:
            termination = 'no_applicable_actions'
            break
        action = learner.choose(rng, available)
        d, r = action
        before = monotonic()
        progress(f'iteration {iteration + 1}/{p.iterations}: started')
        decoder = Decoder(net, current.metadata['timetable'], p)
        decoder.upper = upper
        decoder.objective_scales = allocation_scales
        iteration_deadline = before + p.repair_seconds if p.repair_seconds else None
        partial, pending = destroy(decoder, current, d, rng)
        rng.shuffle(pending)
        trial = None
        failed = None
        decode_calls = decoder.calls
        table = decoder.timetable
        if d == 4 or r == 2:
            shifted = shifted_candidates(net, timetable, table, p, rng, getattr(decoder, 'released_positions', None))
            table = next((candidate for candidate in shifted if candidate != decoder.timetable and validate_branch(candidate, timetable, p)), table)
        if validate_branch(table, timetable, p):
            local = Decoder(net, table, p)
            local.objective_scales = allocation_scales
            local.scales = allocation_scales
            local.tradeoff_weights = rng.choice(DIRECTIONS)
            local.deadline = iteration_deadline
            if search_deadline is not None:
                local.deadline = min(local.deadline or search_deadline, search_deadline)
            from .constructive_repair import compatible
            backend = 'constructive' if p.repair_mode == 'constructive' and compatible(local, partial) else 'exhaustive'
            progress(f'  repair={backend}, pending={len(pending)}, budget={p.repair_seconds}s')
            try:
                trial, failed = repair(local, partial, pending, r, rng)
            except (Infeasible, SearchTimeout) as error:
                failed = str(error)
                if search_deadline is not None and monotonic() >= search_deadline:
                    termination = 'time_budget'
                    break
            decode_calls += local.calls
            if trial is not None and trial.errors:
                trial = None
        gain = improvement = reward = 0.0
        retained = False
        if trial is not None:
            trial.metadata.update(iteration=iteration + 1, iteration_origin='completed_destroy_repair', branch_base=timetable)
            trial.errors = validate(trial, net, None, p)
            if trial.errors:
                trial = None
        if trial is not None:
            old = hypervolume([s.objectives for s in global_archive.items], upper, p.hv_margin)
            new = hypervolume([s.objectives for s in global_archive.items] + [trial.objectives], upper, p.hv_margin)
            gain = max(0, new - old)
            if dominates(trial.objectives, current.objectives):
                improvement = sum(((a - b) / u for a, b, u in zip(current.objectives, trial.objectives, upper)))
            reward = 1.0 if gain > 1e-15 else p.dominance_reward * improvement / (1 + improvement)
            retained = archive.add(trial)
            global_archive.add(trial)
            if checkpoint:
                checkpoint([trial], {'phase': 'iteration_candidate', 'iteration': iteration + 1})
        learner.feedback(action, reward, feasible=trial is not None, hv_gain=gain)
        stall = 0 if retained else stall + 1
        trace.append({'iteration': iteration + 1, 'hv': hypervolume([s.objectives for s in archive.items], upper, p.hv_margin), 'action': action, 'search_state': learner.last_state, 'reward': reward, 'hv_gain': gain, 'pareto_improvement': improvement, 'repair_complete': trial is not None, 'retained': retained, 'served': len(trial.services) if trial else 0, 'trial_objectives': trial.objectives if trial else None, 'before_objectives': current.objectives, 'failed_request': failed, 'destroyed_requests': pending, 'decode_calls': decode_calls, 'repair_backend': backend if validate_branch(table, timetable, p) else None, 'joint_solver_calls': local.optimizer.solves if validate_branch(table, timetable, p) else 0, 'decode_cache_hits': local.decode_cache_hits if validate_branch(table, timetable, p) else 0, 'repair_timed_out': isinstance(failed, str) and 'time budget' in failed.lower(), 'tradeoff_weights': list(local.tradeoff_weights) if validate_branch(table, timetable, p) else None, 'elapsed_seconds': monotonic() - before})
        if checkpoint:
            checkpoint(archive.items, {'phase': 'search', 'iteration': iteration + 1, 'state': state(iteration + 1)})
        progress(f'iteration {iteration + 1}/{p.iterations}: feasible={trial is not None}, reward={reward:.6g}, archive={len(archive.items)}, elapsed={trace[-1]["elapsed_seconds"]:.2f}s')
    return archive.items, {
        'iterations': trace[-1]['iteration'] if trace else offset,
        'complete_feasible_iterations': sum(row['repair_complete'] for row in trace),
        'trace': trace, 'initial_failures': failures, 'termination': termination,
        'elapsed_seconds': previous_elapsed + monotonic() - start,
        'initialization_elapsed_seconds': initialization_elapsed,
        'initialization_construction_seconds': initialization_construction_seconds,
        'initialization_decode_seconds': initialization_decode_seconds,
        'initialization_attempts': initialization_attempts,
        'iteration_elapsed_seconds': sum(row['elapsed_seconds'] for row in trace),
        'learner': learner.snapshot(), 'objective_bounds': upper,
    }
