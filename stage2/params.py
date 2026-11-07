from dataclasses import dataclass

@dataclass(frozen=True)
class Params:
    capacity: int = 6
    max_formation: int = 4
    speed: float = 30
    purchase_cost: float = 50
    operating_cost: float = 0.3
    following_ratio: float = 0.8
    max_detour: float = 15
    max_coupling_wait: float = 10
    depot_capacity: int = 20
    depot_initial: int = 5
    stop_time: float = 20 / 60
    operation_time: float = 30 / 60
    iterations: int = 200
    stall: int = 60
    learning_rate: float = 0.1
    destroy_fraction: float = 0.1
    destroy_fraction_max: float = 0.3
    initial_solutions: int = 3
    initial_relief_reserve: int = 2
    repair_mode: str = 'constructive'
    initial_feasible_target: int | None = None
    initialization: str = 'constructive'
    search_seconds: float | None = None
    initial_checks: int = 30000
    repair_checks: int = 10000
    initial_backtracks: int = 3
    source_limit: int | None = None
    insertion_limit: int | None = None
    join_limit: int | None = None
    pool_limit: int | None = None
    subproblem_seconds: float | None = None
    seed: int = 2026
    conditional_operations: bool = True
    start: float = 480
    end: float = 600
    min_headway: float = 5
    max_headway: float = 15
    shift_limit: float = 2
    repair_budget: int = 20
    repair_seconds: float | None = None
    exploration: float = 0.05
    dominance_reward: float = 0.3
    hv_margin: float = 0.1
    model_version: str = 'symmetric-chebyshev-original-init-v3'

    @property
    def coupling_discount(self):

        return 1 - self.following_ratio

    @property
    def time_value(self):


        return 1.0

    def __post_init__(self):
        if not 0 <= self.depot_initial <= self.depot_capacity:
            raise ValueError('Invalid initial warehouse stock')
        if self.capacity < 1 or self.max_formation < 1 or self.depot_capacity < 1:
            raise ValueError('Invalid capacity')
        if self.speed <= 0 or not 0 <= self.following_ratio <= 1:
            raise ValueError('Invalid speed/cost ratio')
        if min(self.purchase_cost, self.operating_cost, self.stop_time, self.operation_time, self.max_coupling_wait, self.max_detour) < 0:
            raise ValueError('Invalid operational value')
        if not 0 < self.min_headway <= self.max_headway or self.end <= self.start or self.shift_limit < 0:
            raise ValueError('Invalid timetable bounds')
        if not 0 < self.destroy_fraction <= self.destroy_fraction_max <= 1:
            raise ValueError('Invalid destroy fraction range')
        if self.initial_checks < 1 or self.repair_checks < 1 or self.initial_backtracks < 0:
            raise ValueError('Invalid initialization search budget')
        if self.iterations < 0 or min(self.initial_solutions, self.repair_budget, self.stall) < 1:
            raise ValueError('Invalid search budget')
        if not 0 < self.learning_rate < 1 or not 0 < self.dominance_reward < 1 or self.hv_margin <= 0:
            raise ValueError('Invalid learning value')
        if not 0 <= self.exploration <= 1:
            raise ValueError('Invalid exploration')
        if self.repair_mode not in ('constructive', 'exhaustive') or self.initialization not in ('constructive', 'insertion'):
            raise ValueError('Unknown initialization/repair policy')
        if self.initial_relief_reserve < 0 or (self.initial_feasible_target is not None and self.initial_feasible_target < 1):
            raise ValueError('Invalid legacy initialization setting')
        for value in (self.subproblem_seconds, self.repair_seconds, self.search_seconds):
            if value is not None and value <= 0:
                raise ValueError('Invalid solve time limit')
