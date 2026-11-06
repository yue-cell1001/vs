from .operators import remove
from .routing import Infeasible


def prepare(decoder, initial, pending):
    plan = tuple(initial)
    pending = set(pending)
    while plan:
        try:
            decoder.context(plan)
            break
        except Infeasible:
            invalid = None
            for index, item in enumerate(plan):
                try:
                    decoder.context(plan[:index + 1])
                except Infeasible:
                    invalid = item.source
                    break
            if invalid is None:
                raise
            old = {pid for e in plan for pid, _ in e.events}
            plan = remove(plan, (), {invalid})
            pending.update(old - {pid for e in plan for pid, _ in e.events})
    return list(plan), pending
