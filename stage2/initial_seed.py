from collections import Counter
import numpy as np
from common.integer import _verified


def reservation_seed(hint, groups, members, columns, requests, use, birth, finish, chosen, lower, upper, matrix, row_lower, row_upper):
    if not hint:
        return None
    x = np.zeros(len(lower))
    membership = {name: key for key, names in members.items() for name in names}
    for key in groups:
        amount = sum(name in hint['active'] for name in members[key])
        x[use[key]] = x[birth[key]] = x[finish[key]] = amount
    desired = {pid: (membership[key], a, b) for pid, (key, a, b) in hint['ordinary_rides'].items() if key in membership}
    for count, options in requests:
        cohort = columns[options[0]].people
        counts = Counter()
        for pid in cohort:
            if pid not in desired:
                counts[options[0]] += 1
                continue
            target = desired[pid]
            selected = next((index for index in options if len(columns[index].rides) == 1 and (columns[index].rides[0].duty, columns[index].rides[0].board, columns[index].rides[0].alight) == target), None)
            if selected is None:
                return None
            counts[selected] += 1
        for index, amount in counts.items():
            x[chosen[index]] += amount
    return _verified(x, lower, upper, matrix, row_lower, row_upper)
