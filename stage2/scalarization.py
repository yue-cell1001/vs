import numpy as np
from scipy.sparse import csr_array, hstack, vstack

DIRECTIONS = ((.2, .8), (.35, .65), (.5, .5), (.65, .35), (.8, .2))


def positive_pair(values):
    values = tuple(float(x) for x in values)
    if len(values) != 2 or any(not np.isfinite(x) or x <= 0 for x in values):
        raise ValueError('Invalid objective scales or weights')
    return values


def reference_scales(candidates, fallback):
    points = [item.get('objectives') for item in candidates]
    if not points or any(point is None or len(point) != 2 for point in points):
        return positive_pair(fallback)
    if any(not np.isfinite(x) or x < 0 for point in points for x in point):
        raise ValueError('Invalid reference objectives')
    return tuple(max(1.0, max(point[j] for point in points)) for j in (0, 1))


def build_problem(c1, c2, constant, lower, upper, matrix, lb, ub, seed, scales, weights):
    scales = positive_pair(scales)
    weights = positive_pair(weights)
    total = sum(weights)
    weights = tuple(w / total for w in weights)
    a = np.asarray(c1, float) * (1e6 * weights[0] / scales[0])
    b = np.asarray(c2, float) * (1e6 * weights[1] / scales[1])
    offset = 1e6 * weights[0] * constant / scales[0]
    n = len(c1)
    core = hstack([matrix, csr_array((matrix.shape[0], 1))], format='csc')
    extra = csr_array(np.vstack([np.r_[a, -1.0], np.r_[b, -1.0]]))
    full = vstack([core, extra], format='csc')
    warm = None if seed is None else np.r_[seed, max(float(a @ seed + offset), float(b @ seed), 0.0)]
    return (np.r_[.001 * (a + b), 1.0], [*lower, 0.0], [*upper, np.inf], full,
            [*lb, -np.inf, -np.inf], [*ub, -offset, 0.0], warm, [1] * n + [0], weights)
