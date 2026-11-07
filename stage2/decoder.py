from copy import deepcopy
from common.model import Solution
from common.timetable import make_trips
from .base import build_base
from .routing import construct_duties, Infeasible
from .constraints import validate
from .obj import evaluate

class Decoder:

    def __init__(self, net, timetable, p):
        self.net, self.p = (net, p)
        self.timetable = timetable
        self.trips = make_trips(net, timetable, p)
        self.base = build_base(net, self.trips, p)
        self.nominal_trips = self.trips
        self.nominal_base = self.base
        self.contexts = {}
        self.calls = 0
        self.decode_cache = {}
        self.decode_cache_hits = 0
        self.people = {x.id: x for x in net.passengers}
        from .joint import JointOptimizer
        self.optimizer = JointOptimizer(net, p)
        self.preference = 0
        self.objective_scales = None
        self.tradeoff_weights = (.5, .5)
        self.allow_solo = True
        from .incremental import DeltaAudit, DeltaEvaluator
        self.delta_audit = DeltaAudit(net, p)
        self.delta_evaluator = DeltaEvaluator(net, p)

    def context(self, excursions):
        from .budget import check
        check(self)
        from .context import compile_context
        key = tuple(excursions)
        if key not in self.contexts:
            self.contexts[key] = compile_context(self, key)
            if len(self.contexts) > 16:
                self.contexts.pop(next(iter(self.contexts)))
        return self.contexts[key]

    def decode(self, excursions, required=None, audit=False):
        self.calls += 1
        if required is None:
            required = {pid for e in excursions for pid, _ in e.events}
        required = set(required)
        cache_key = (tuple(excursions), frozenset(required), self.preference, tuple(self.tradeoff_weights), tuple(self.objective_scales) if self.objective_scales is not None else None, bool(audit), bool(getattr(self.optimizer, 'verify_only', False)))
        try:
            context = self.context(excursions)
            self.trips, self.base = (context.trips, context.base)
            if audit and cache_key in self.decode_cache:
                self.decode_cache_hits += 1
                return deepcopy(self.decode_cache[cache_key])
            duties = context.duties
            operations = context.operations
            if self.objective_scales is not None:
                self.optimizer.objective_scales = tuple(self.objective_scales)
            self.optimizer.tradeoff_weights = tuple(self.tradeoff_weights)
            solution = self.optimizer.solve(duties, self.trips, excursions, required, self.preference)
            solution.metadata['allocation_preference'] = self.preference
            solution.metadata['timetable'] = self.timetable
            solution.metadata['operations'] = [list(action) for action in sorted(operations)]
            solution.metadata['scheduling_actions'] = context.actions
            self.delta_audit.check(solution)
            self.delta_evaluator.update(solution)
            if audit:
                solution.errors = validate(solution, self.net, self.trips, self.p, required)
                if solution.errors:
                    raise Infeasible('; '.join(solution.errors[:4]))
            numeric = solution.metadata['joint_solver']
            if abs(solution.objectives[0] - numeric['F1_linear']) > 1e-05 or abs(solution.objectives[1] - numeric['F2_linear']) > 1e-05:
                raise Infeasible('Joint integer model / physical objective mismatch')
            if audit:
                self.decode_cache[cache_key] = deepcopy(solution)
                if len(self.decode_cache) > 4:
                    self.decode_cache.pop(next(iter(self.decode_cache)))
            return solution
        except Infeasible as error:
            return Solution({}, {}, tuple(excursions), errors=[str(error)])
