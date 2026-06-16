import numpy as np
from typing import Callable, Union

ArrayLike = Union[float, np.ndarray]


def central_difference(
    f: Callable[[ArrayLike], ArrayLike],
    x: ArrayLike,
    h: float = 1e-5,
) -> ArrayLike:
    return (f(x + h) - f(x - h)) / (2 * h)


def numerical_gradient(
    f: Callable[[ArrayLike], ArrayLike],
    x: np.ndarray,
    h: float = 1e-5,
) -> np.ndarray:
    grad = np.zeros_like(x, dtype=float)
    for i in range(x.size):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus.flat[i] += h
        x_minus.flat[i] -= h
        grad.flat[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad


class NumericalDifferentiator:
    def __init__(self, h: float = 1e-5):
        if h <= 0:
            raise ValueError(f"Step size h must be positive, got {h}")
        self.h = h

    def derivative(
        self,
        f: Callable[[ArrayLike], ArrayLike],
        x: ArrayLike,
    ) -> ArrayLike:
        return central_difference(f, x, self.h)

    def gradient(
        self,
        f: Callable[[np.ndarray], ArrayLike],
        x: np.ndarray,
    ) -> np.ndarray:
        return numerical_gradient(f, x, self.h)

    def second_derivative(
        self,
        f: Callable[[ArrayLike], ArrayLike],
        x: float,
    ) -> float:
        h = self.h
        return (f(x + h) - 2 * f(x) + f(x - h)) / (h * h)

    def partial_derivative(
        self,
        f: Callable[[np.ndarray], ArrayLike],
        x: np.ndarray,
        var_index: int,
    ) -> float:
        h = self.h
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[var_index] += h
        x_minus[var_index] -= h
        return float((f(x_plus) - f(x_minus)) / (2 * h))
