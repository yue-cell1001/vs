from dataclasses import dataclass
from math import isfinite
import numpy as np
import warnings
from highspy import Highs, HighsModelStatus, HighsVarType

@dataclass
class Result:
    x: np.ndarray | None
    optimal: bool
    gap: float | None
    lower_bound: float | None
    message: str

def _verified(x, lower, upper, matrix, row_lower, row_upper, integrality=None):
    if x is None:
        return None
    x = np.asarray(x, float)
    if x.shape != (len(lower),):
        return None
    mask = np.ones(len(lower), dtype=bool) if integrality is None else np.asarray(integrality, dtype=bool)
    if mask.shape != x.shape:
        return None
    rounded = x.copy()
    rounded[mask] = np.rint(x[mask])
    residual = matrix @ rounded
    if not np.all(np.isfinite(x)) or np.max(np.abs(x - rounded), initial=0) > 1e-05 or np.any(rounded < np.asarray(lower) - 1e-05) or np.any(rounded > np.asarray(upper) + 1e-05) or np.any(residual < np.asarray(row_lower) - 1e-05) or np.any(residual > np.asarray(row_upper) + 1e-05):
        return None
    return rounded

def solve_integer(cost, lower, upper, matrix, row_lower, row_upper, seconds=None, warm=None, integrality=None):
    try:
        return _solve_integer(cost, lower, upper, matrix, row_lower, row_upper, seconds, warm, integrality)
    except MemoryError:
        x = _verified(warm, lower, upper, matrix, row_lower, row_upper, integrality)
        if x is None:
            raise MemoryError('HiGHS ran out of memory; no verified incumbent is available. Close other applications or increase Windows virtual memory.') from None
        warnings.warn('HiGHS memory allocation failed; using the fully constraint-verified warm incumbent. Optimality is not proved.', RuntimeWarning, stacklevel=2)
        return Result(x, False, None, None, 'memory_limit_verified_warm_incumbent')

def _solve_integer(cost, lower, upper, matrix, row_lower, row_upper, seconds, warm, integrality):
    h = Highs()
    try:
        h.setOptionValue('output_flag', False)
        h.setOptionValue('threads', 1)
        h.setOptionValue('mip_rel_gap', 0.0)
        if seconds is not None:
            h.setOptionValue('time_limit', float(seconds))
        n = len(cost)
        csr = matrix.tocsr()
        h.addCols(n, np.asarray(cost, float), np.asarray(lower, float), np.asarray(upper, float), 0, np.zeros(n + 1, np.int32), np.empty(0, np.int32), np.empty(0, float))
        h.addRows(matrix.shape[0], np.asarray(row_lower, float), np.asarray(row_upper, float), csr.nnz, csr.indptr.astype(np.int32), csr.indices.astype(np.int32), csr.data.astype(float))
        kinds = [HighsVarType.kInteger] * n if integrality is None else [HighsVarType.kInteger if flag else HighsVarType.kContinuous for flag in integrality]
        if len(kinds) != n:
            raise ValueError('Invalid integrality mask')
        h.changeColsIntegrality(n, np.arange(n, dtype=np.int32), kinds)
        if warm is not None:
            h.setSolution(n, np.arange(n, dtype=np.int32), np.asarray(warm, float))
        h.run()
        status = h.getModelStatus()
        solution = h.getSolution()
        info = h.getInfo()
        x = _verified(solution.col_value if solution.value_valid else None, lower, upper, matrix, row_lower, row_upper, integrality)

        def finite(value):
            return float(value) if isfinite(value) else None
        return Result(x, status == HighsModelStatus.kOptimal, finite(info.mip_gap), finite(info.mip_dual_bound), h.modelStatusToString(status))
    finally:
        h.clear()
