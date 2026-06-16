import numpy as np
from typing import Callable, Union, Tuple

ArrayLike = Union[float, np.ndarray]

_EPS = np.finfo(float).eps
_DEFAULT_H_CANDIDATES = np.logspace(-2, -10, 9)
_H_CANDIDATES_2ND = np.logspace(-2, -7, 6)


def _robust_error_estimate(
    results: list,
    order: int = 2,
) -> int:
    if len(results) < 3:
        return 0

    fac = 1.0 / (2**order - 1.0)
    errors = []
    for i in range(len(results) - 1):
        d1 = results[i][1]
        d2 = results[i + 1][1]
        if isinstance(d1, np.ndarray):
            err = float(np.mean(np.abs(fac * (d2 - d1))))
        else:
            err = float(abs(fac * (d2 - d1)))
        errors.append(err)

    min_err_idx = int(np.argmin(errors))

    if len(errors) >= 3:
        for i in range(1, len(errors) - 1):
            if errors[i] < errors[i - 1] and errors[i] < errors[i + 1]:
                if errors[i + 1] > errors[i] * 2 and errors[i] < 1e-8:
                    return i

        if min_err_idx >= len(errors) - 2:
            for i in range(len(errors) - 3, -1, -1):
                if errors[i] < errors[i + 1]:
                    return i

    if min_err_idx == len(errors) - 1 and len(errors) >= 3:
        if errors[-1] > errors[-2] and errors[-2] > errors[-3]:
            return len(errors) - 3

    return min_err_idx


def _check_convergence(
    results: list,
    order: int = 2,
    min_results: int = 3,
) -> Tuple[int, bool]:
    if len(results) < min_results:
        return len(results) - 1, False

    fac = 1.0 / (2**order - 1.0)
    errors = []
    for i in range(len(results) - 1):
        d1 = results[i][1]
        d2 = results[i + 1][1]
        if isinstance(d1, np.ndarray):
            err = float(np.mean(np.abs(fac * (d2 - d1))))
        else:
            err = float(abs(fac * (d2 - d1)))
        errors.append(err)

    min_err_idx = int(np.argmin(errors))
    converged = False

    if min_err_idx > 0 and min_err_idx < len(errors) - 1:
        if errors[min_err_idx] < errors[min_err_idx - 1] * 0.5:
            converged = True

    return min_err_idx, converged


def _cd_core(
    f: Callable[[ArrayLike], ArrayLike],
    x: ArrayLike,
    h: float,
) -> ArrayLike:
    return (f(x + h) - f(x - h)) / (2 * h)


def _cd2_core(
    f: Callable[[ArrayLike], ArrayLike],
    x: ArrayLike,
    h: float,
) -> ArrayLike:
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h * h)


def _estimate_error(
    d_h: ArrayLike,
    d_h2: ArrayLike,
    order: int = 2,
) -> ArrayLike:
    fac = 1.0 / (2**order - 1.0)
    return fac * np.abs(d_h2 - d_h)


def _auto_step_first_order(
    f: Callable[[ArrayLike], ArrayLike],
    x: ArrayLike,
    h_candidates: np.ndarray,
) -> Tuple[ArrayLike, float]:
    results = []
    for h in h_candidates:
        try:
            d = _cd_core(f, x, h)
            results.append((h, d))
        except (ZeroDivisionError, FloatingPointError, ValueError):
            continue

    if len(results) < 2:
        h = h_candidates[len(h_candidates) // 2]
        return _cd_core(f, x, h), h

    best_idx = _robust_error_estimate(results, order=2)
    best_idx = min(best_idx, len(results) - 2)
    return results[best_idx + 1][1], results[best_idx + 1][0]


def _auto_step_second_order(
    f: Callable[[ArrayLike], ArrayLike],
    x: ArrayLike,
    h_candidates: np.ndarray = _H_CANDIDATES_2ND,
) -> Tuple[ArrayLike, float]:
    results = []
    for h in h_candidates:
        try:
            d = _cd2_core(f, x, h)
            results.append((h, d))
        except (ZeroDivisionError, FloatingPointError, ValueError):
            continue

    if len(results) < 2:
        h = h_candidates[len(h_candidates) // 2]
        return _cd2_core(f, x, h), h

    best_idx = _robust_error_estimate(results, order=2)
    best_idx = min(best_idx, len(results) - 2)
    return results[best_idx + 1][1], results[best_idx + 1][0]


def _resolve_h(h: Union[float, str], default: float) -> Union[float, str]:
    if isinstance(h, str):
        if h.lower() == 'auto':
            return 'auto'
        raise ValueError(f"Invalid string h='{h}'. Use 'auto' or a positive float.")
    if h <= 0:
        raise ValueError(f"Step size h must be positive, got {h}")
    return h


def central_difference(
    f: Callable[[ArrayLike], ArrayLike],
    x: ArrayLike,
    h: Union[float, str] = 'auto',
    return_h: bool = False,
    h_candidates: np.ndarray = _DEFAULT_H_CANDIDATES,
) -> Union[ArrayLike, Tuple[ArrayLike, float]]:
    h_resolved = _resolve_h(h, 1e-5)
    if h_resolved == 'auto':
        d, best_h = _auto_step_first_order(f, x, h_candidates)
        return (d, best_h) if return_h else d
    else:
        d = _cd_core(f, x, h_resolved)
        return (d, h_resolved) if return_h else d


def numerical_gradient(
    f: Callable[[ArrayLike], ArrayLike],
    x: np.ndarray,
    h: Union[float, str] = 'auto',
    h_candidates: np.ndarray = _DEFAULT_H_CANDIDATES,
) -> np.ndarray:
    h_resolved = _resolve_h(h, 1e-5)
    grad = np.zeros_like(x, dtype=float)

    if h_resolved == 'auto':
        for i in range(x.size):
            def f_i(t: float, _i: int = i) -> float:
                xi = x.copy()
                xi.flat[_i] = t
                return float(f(xi))
            d, _ = _auto_step_first_order(f_i, float(x.flat[i]), h_candidates)
            grad.flat[i] = float(d)
    else:
        for i in range(x.size):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus.flat[i] += h_resolved
            x_minus.flat[i] -= h_resolved
            grad.flat[i] = (f(x_plus) - f(x_minus)) / (2 * h_resolved)
    return grad


def central_difference_2nd(
    f: Callable[[ArrayLike], ArrayLike],
    x: ArrayLike,
    h: Union[float, str] = 'auto',
    return_h: bool = False,
    h_candidates: np.ndarray = _H_CANDIDATES_2ND,
) -> Union[ArrayLike, Tuple[ArrayLike, float]]:
    h_resolved = _resolve_h(h, 1e-4)
    if h_resolved == 'auto':
        d, best_h = _auto_step_second_order(f, x, h_candidates)
        return (d, best_h) if return_h else d
    else:
        d = _cd2_core(f, x, h_resolved)
        return (d, h_resolved) if return_h else d


def numerical_hessian(
    f: Callable[[np.ndarray], ArrayLike],
    x: np.ndarray,
    h: Union[float, str] = 'auto',
    h_candidates_1st: np.ndarray = _DEFAULT_H_CANDIDATES,
    h_candidates_2nd: np.ndarray = _H_CANDIDATES_2ND,
) -> np.ndarray:
    n = x.size
    hess = np.zeros((n, n), dtype=float)
    h_resolved = _resolve_h(h, 1e-4)

    for i in range(n):
        for j in range(i, n):
            if i == j:
                def f_ii(t: float, _i: int = i) -> float:
                    xi = x.copy()
                    xi.flat[_i] = t
                    return float(f(xi))
                if h_resolved == 'auto':
                    d, _ = _auto_step_second_order(f_ii, float(x.flat[i]), h_candidates_2nd)
                    hess[i, i] = float(d)
                else:
                    t = float(x.flat[i])
                    hess[i, i] = float(
                        (f_ii(t + h_resolved) - 2 * f_ii(t) + f_ii(t - h_resolved))
                        / (h_resolved * h_resolved)
                    )
            else:
                def f_ij(s: float, t: float, _i: int = i, _j: int = j) -> float:
                    xi = x.copy()
                    xi.flat[_i] = s
                    xi.flat[_j] = t
                    return float(f(xi))

                xi = float(x.flat[i])
                xj = float(x.flat[j])

                if h_resolved == 'auto':
                    h_use = h_candidates_2nd[len(h_candidates_2nd) // 2]
                else:
                    h_use = float(h_resolved)

                hi = h_use
                hj = h_use

                hess[i, j] = (
                    f_ij(xi + hi, xj + hj)
                    - f_ij(xi + hi, xj - hj)
                    - f_ij(xi - hi, xj + hj)
                    + f_ij(xi - hi, xj - hj)
                ) / (4 * hi * hj)
                hess[j, i] = hess[i, j]

    return hess


class NumericalDifferentiator:
    def __init__(
        self,
        h: Union[float, str] = 'auto',
        h_candidates: np.ndarray = _DEFAULT_H_CANDIDATES,
        h_candidates_2nd: np.ndarray = _H_CANDIDATES_2ND,
    ):
        h_resolved = _resolve_h(h, 1e-5)
        self.h = h_resolved
        self.h_candidates = h_candidates
        self.h_candidates_2nd = h_candidates_2nd

    def derivative(
        self,
        f: Callable[[ArrayLike], ArrayLike],
        x: ArrayLike,
        return_h: bool = False,
    ) -> Union[ArrayLike, Tuple[ArrayLike, float]]:
        return central_difference(f, x, self.h, return_h=return_h, h_candidates=self.h_candidates)

    def gradient(
        self,
        f: Callable[[np.ndarray], ArrayLike],
        x: np.ndarray,
    ) -> np.ndarray:
        return numerical_gradient(f, x, self.h, h_candidates=self.h_candidates)

    def second_derivative(
        self,
        f: Callable[[ArrayLike], ArrayLike],
        x: ArrayLike,
        return_h: bool = False,
    ) -> Union[ArrayLike, Tuple[ArrayLike, float]]:
        return central_difference_2nd(f, x, self.h, return_h=return_h, h_candidates=self.h_candidates_2nd)

    def hessian(
        self,
        f: Callable[[np.ndarray], ArrayLike],
        x: np.ndarray,
    ) -> np.ndarray:
        return numerical_hessian(f, x, self.h, h_candidates_1st=self.h_candidates, h_candidates_2nd=self.h_candidates_2nd)

    def partial_derivative(
        self,
        f: Callable[[np.ndarray], ArrayLike],
        x: np.ndarray,
        var_index: int,
        return_h: bool = False,
    ) -> Union[float, Tuple[float, float]]:
        def f_i(t: float) -> float:
            xi = x.copy()
            xi[var_index] = t
            return float(f(xi))
        return central_difference(f_i, float(x[var_index]), self.h, return_h=return_h, h_candidates=self.h_candidates)

    def second_partial_derivative(
        self,
        f: Callable[[np.ndarray], ArrayLike],
        x: np.ndarray,
        var_index_1: int,
        var_index_2: int,
    ) -> float:
        if var_index_1 == var_index_2:
            def f_ii(t: float) -> float:
                xi = x.copy()
                xi[var_index_1] = t
                return float(f(xi))
            return float(central_difference_2nd(f_ii, float(x[var_index_1]), self.h, h_candidates=self.h_candidates_2nd))
        else:
            h_resolved = self.h
            if h_resolved == 'auto':
                h_use = self.h_candidates_2nd[len(self.h_candidates_2nd) // 2]
            else:
                h_use = float(h_resolved)
            xi = float(x[var_index_1])
            xj = float(x[var_index_2])
            h1 = h_use
            h2 = h_use

            def f_ij(s: float, t: float) -> float:
                xi_copy = x.copy()
                xi_copy[var_index_1] = s
                xi_copy[var_index_2] = t
                return float(f(xi_copy))

            return float(
                (f_ij(xi + h1, xj + h2)
                 - f_ij(xi + h1, xj - h2)
                 - f_ij(xi - h1, xj + h2)
                 + f_ij(xi - h1, xj - h2))
                / (4 * h1 * h2)
            )
