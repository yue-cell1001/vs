from dataclasses import dataclass, field, asdict
from typing import Any

@dataclass(frozen=True)
class Node:
    id: int
    kind: int
    name: str
    x: float
    y: float

@dataclass(frozen=True)
class Passenger:
    id: int
    origin: int
    dest: int
    kind: int
    earliest: float
    latest: float
    line: str
    origin_transfer: int | None = None
    destination_transfer: int | None = None

@dataclass
class Network:
    nodes: dict[int, Node]
    lines: dict[str, tuple[int, ...]]
    edges: dict[tuple[int, int], tuple[float, float]]
    passengers: list[Passenger]
    depots: dict[int, int]
    line_families: dict[str, str] = field(default_factory=dict)
    transfer_candidates: dict[int, dict] = field(default_factory=dict)

    def distance(self, a: int, b: int) -> float:
        from math import hypot
        x, y = (self.nodes[a], self.nodes[b])
        return hypot(x.x - y.x, x.y - y.y) / 1000.0

    def road(self, a: int, b: int) -> tuple[float, float]:
        return self.edges[a, b]

@dataclass(frozen=True)
class Visit:
    node: int
    arrival: float
    departure: float
    trip: str = ''
    line: str = ''
    kind: str = 'fixed'

@dataclass(frozen=True)
class Leg:
    start: int
    end: int
    departure: float
    arrival: float
    distance: float
    kind: str
    trip: str = ''
    line: str = ''

@dataclass
class Duty:
    id: str
    visits: list[Visit]
    legs: list[Leg]
    vehicle: int = -1

@dataclass(frozen=True)
class Ride:
    duty: str
    board: int
    alight: int

@dataclass(frozen=True)
class Service:
    passenger: int
    rides: tuple[Ride, ...]
    pickup: float
    dropoff: float
    detour: float = 0.0
    offroute_minutes: float = 0.0

@dataclass(frozen=True)
class Excursion:
    source: str
    split: int
    target: str
    join: int
    events: tuple[tuple[int, bool], ...]
    split_trip: str = ''
    split_node: int | None = None
    release_time: float | None = None

@dataclass
class Solution:
    duties: dict[str, Duty]
    services: dict[int, Service]
    excursions: tuple[Excursion, ...]
    objectives: tuple[float, float] = (float('inf'), float('inf'))
    components: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def signature(self):
        from collections import defaultdict
        chains = defaultdict(list)
        riders = defaultdict(list)
        for key, duty in self.duties.items():
            path = tuple(((v.node, round(v.arrival, 7), round(v.departure, 7), v.trip, v.line, v.kind) for v in duty.visits))
            chains[duty.vehicle].append(path)
        for pid, service in self.services.items():
            for ride in service.rides:
                riders[self.duties[ride.duty].vehicle].append(pid)
        order = sorted(chains, key=lambda vehicle: (tuple(sorted(chains[vehicle])), tuple(sorted(riders[vehicle]))))
        numbered = {vehicle: i for i, vehicle in enumerate(order)}
        paths = tuple((tuple(sorted(chains[v])) for v in order))
        services = []
        for pid, service in sorted(self.services.items()):
            rides = []
            for r in service.rides:
                d = self.duties[r.duty]
                rides.append((numbered[d.vehicle], d.visits[r.board].node, round(d.visits[r.board].departure, 7), d.visits[r.alight].node, round(d.visits[r.alight].arrival, 7)))
            services.append((pid, tuple(rides)))
        return (paths, tuple(services))

    def to_dict(self):
        return asdict(self)
