from math import inf

def objective_key(value):
    return tuple((round(float(x), 7) for x in value))

def dominates(a, b):
    a, b = (objective_key(a), objective_key(b))
    return all((x <= y for x, y in zip(a, b))) and any((x < y for x, y in zip(a, b)))

def fronts(values):
    count = [0] * len(values)
    successors = [[] for _ in values]
    for i, a in enumerate(values):
        for j in range(i + 1, len(values)):
            b = values[j]
            if dominates(a, b):
                successors[i].append(j)
                count[j] += 1
            elif dominates(b, a):
                successors[j].append(i)
                count[i] += 1
    front = [i for i, c in enumerate(count) if c == 0]
    result = []
    while front:
        result.append(front)
        following = []
        for i in front:
            for j in successors[i]:
                count[j] -= 1
                if count[j] == 0:
                    following.append(j)
        front = following
    return result

def crowding(values):
    values = [objective_key(v) for v in values]
    n = len(values)
    out = [0.0] * n
    if not n:
        return out
    for m in range(2):
        order = sorted(range(n), key=lambda i: values[i][m])
        span = values[order[-1]][m] - values[order[0]][m]
        if span:
            out[order[0]] = out[order[-1]] = inf
            for k in range(1, n - 1):
                out[order[k]] += (values[order[k + 1]][m] - values[order[k - 1]][m]) / span
    return out

def select(values, size):
    keep = []
    for front in fronts(values):
        if len(keep) + len(front) <= size:
            keep.extend(front)
        else:
            distance = crowding([values[i] for i in front])
            keep.extend(sorted(front, key=lambda i: distance[front.index(i)], reverse=True)[:size - len(keep)])
            break
    return keep

class Archive:

    def __init__(self, capacity=None):
        self.capacity = capacity
        self.items = []
        self._fingerprints = {}

    def add(self, solution):
        if solution.errors or any((dominates(s.objectives, solution.objectives) for s in self.items)):
            return False
        signature = solution.signature()
        fingerprint = hash(signature)
        for old in self.items:
            if id(old) not in self._fingerprints:
                self._fingerprints[id(old)] = hash(old.signature())
            if self._fingerprints[id(old)] == fingerprint and old.signature() == signature:
                return False
        pool = [s for s in self.items if not dominates(solution.objectives, s.objectives)] + [solution]
        while self.capacity is not None and len(pool) > self.capacity:
            cd = crowding([s.objectives for s in pool])
            pool.pop(min(range(len(pool)), key=lambda i: cd[i]))
        self.items = pool
        self._fingerprints[id(solution)] = fingerprint
        self._fingerprints = {id(s): self._fingerprints[id(s)] for s in pool}
        return any((s is solution for s in pool))
