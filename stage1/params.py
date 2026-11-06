from dataclasses import dataclass

@dataclass(frozen=True)
class Params:
    start: float = 480
    end: float = 600
    min_headway: float = 5
    max_headway: float = 15
    population: int = 64
    generations: int = 200
    evaluation_budget: int | None = None
    crossover: float = 0.9
    mutation_step: float = 2
    seed: int = 2026
    capacity: int = 6
    max_formation: int = 4
    purchase_cost: float = 50
    operating_cost: float = 0.3
    following_ratio: float = 0.8
    reference_load: float = 0.8
    depot_initial: int = 5
    depot_capacity: int = 20
    turnaround_time: float = 1.0
    stop_time: float = 20 / 60
    operation_time: float = 30 / 60
    candidates: int = 20
    guided_fraction: float = 0.5
    demand_bin: float = 5
    conditional_operations: bool = True
    demand_guided_mutation: float = 0.5
    insert_probability: float = 0.1
    delete_probability: float = 0.1
    shift_probability: float = 0.1
    mutation_candidates: int = 10
    repair_budget: int = 20

    def __post_init__(self):
        if not (0 < self.min_headway <= self.max_headway and self.end > self.start):
            raise ValueError('Invalid horizon/headway bounds')
        from math import ceil, floor
        if ceil((self.end - self.start) / self.max_headway) > floor((self.end - self.start) / self.min_headway):
            raise ValueError('No feasible number of headways covers the fixed service horizon')
        if not 0 <= self.depot_initial <= self.depot_capacity:
            raise ValueError('Invalid warehouse stock/capacity')
        if self.capacity < 1 or self.max_formation < 1 or (not 0 < self.reference_load <= 1) or (not 0 <= self.following_ratio <= 1) or (self.turnaround_time < 0):
            raise ValueError('Invalid reference model parameters')
        if not 0 <= self.guided_fraction <= 1 or self.demand_bin <= 0:
            raise ValueError('Invalid demand-guided initialization parameters')
        if self.evaluation_budget is not None and self.evaluation_budget < 1:
            raise ValueError('Invalid evaluation budget')
        if self.population < 2 or self.generations < 0 or (self.candidates is not None and self.candidates < 1):
            raise ValueError('Invalid search size')
