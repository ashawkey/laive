import numpy as np
from .. iteration import solve_iterative


def test_iteration():
    A = np.array([[3.0, 1.0, 0., 0., 0., 0., 0., 0., 0., 0.],[1.0, 3.0, 1.0, 0., 0., 0., 0., 0., 0., 0.], [0., 1.0, 3.0, 1.0, 0., 0., 0., 0., 0., 0.], [0., 0, 1.0, 3.0, 1.0, 0., 0., 0., 0., 0.], [0., 0., 0., 1.0, 3.0, 1.0, 0., 0., 0., 0.], [0., 0., 0., 0., 1.0, 3.0, 1.0, 0., 0., 0.], [0., 0., 0., 0., 0., 1.0, 3.0, 1.0, 0., 0.], [0., 0., 0., 0., 0., 0., 1.0, 3.0, 1.0, 0.], [0., 0., 0., 0., 0., 0., 0., 1.0, 3.0, 1.0], [0., 0., 0., 0., 0., 0., 0., 0., 1.0, 3.0]])
    b = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    print(solve_iterative(A, b, method="Jacobi", verbose=True))
    print(solve_iterative(A, b, method="GS", verbose=True))
    print(solve_iterative(A, b, method="SOR", verbose=True))