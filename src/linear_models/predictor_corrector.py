import numpy as np
import numpy.linalg as linalg
import scipy.linalg as la

def _compute_diagonal(A: np.ndarray) -> np.ndarray:
    return np.array([A[i][i] for i in range(A.shape[1])])

def _update_z(
        P: np.ndarray,
        r: np.ndarray,
        ascending: bool,
        algorithm: str
) -> np.ndarray:

    if algorithm == 'jacobi':
        z = r / P
    elif algorithm == 'gauss_seidel':
        n = r.shape[0]
        z = np.zeros(r.shape)

        if ascending:
            for i in range(n):
                old_sum = np.dot(z[:i], P[i, :i])
                z[i] = (r[i] - old_sum) / P[i, i]
        else:
            # range(start, stop, step)
            # stop is non_inclusive
            for i in range(n - 1, -1, -1):
                old_sum = np.dot(z[i + 1:], P[i, i + 1:])
                z[i] = (r[i] - old_sum) / P[i, i]

    return z

# TODO: add symmetric gauss-seidel
def linear_solver(
        A: np.ndarray,
        b: np.ndarray,
        algorithm: str,
        max_iterations: int=10000,
        w: float=1,
        epsilon: float=1e-14,
        abs_metric: bool=True,   # use the absolute error as the metric
        ascending: bool=True,    # ascending Gauss-Seidel (or not)
) -> None:

    x = np.zeros(b.shape[0])
    r = b - (A @ x)

    if algorithm == 'jacobi':
        P = _compute_diagonal(A) / w
    elif algorithm == 'gauss_seidel':

        _, L, U = la.lu(A)

        if ascending:
            P = _compute_diagonal(A) + L
        else:
            P = _compute_diagonal(A) + U

    iteration = 1
    while iteration < max_iterations + 1:

        print(f"Current iteration = {iteration}")

        z = _update_z(P, r, ascending, algorithm)

        x_next = x + w * z
        r -= w * (A @ z)

        residual_error = linalg.norm(r, ord=2)
        abs_error = linalg.norm(x_next - x) / w

        x = x_next

        print(f"residual_error = {residual_error:.4e}")
        print(f"abs_error = {abs_error:.4e}")

        if abs_metric and abs_error < epsilon:
            break

        if not abs_metric and residual_error < epsilon:
            break

        iteration += 1

    print(f"x = {x}")



