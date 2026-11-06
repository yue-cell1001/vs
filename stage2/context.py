from dataclasses import dataclass
from common.timetable import make_trips
from .base import build_base
from .routing import construct_duties

@dataclass
class Context:
    trips: dict
    base: dict
    duties: dict
    operations: set
    targets: dict
    actions: list

def compile_context(decoder, excursions):
    nominal = decoder.nominal_trips
    base = decoder.nominal_base
    _, operations = construct_duties(decoder.net, nominal, base, excursions, decoder.p, relaxed=True, with_operations=True)
    trips = make_trips(decoder.net, decoder.timetable, decoder.p, operations)
    base = build_base(decoder.net, trips, decoder.p)
    actions = []
    duties = construct_duties(decoder.net, trips, base, excursions, decoder.p, operation_schedule=operations, action_log=actions)
    from .events import verify_context
    from .routing import Infeasible
    try:
        verify_context(decoder.net, decoder.p, trips, duties)
    except ValueError as error:
        raise Infeasible(str(error)) from error
    from .virtual import available_trips
    return Context(trips, base, duties, operations, available_trips(duties, trips), actions)
