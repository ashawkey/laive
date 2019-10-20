import numpy as np
from laive import solve, solve_iterative
from laive.questions import question_4_1
from laive.tests.test_iteration import test_iteration


for epsilon in [1, 0.1, 0.01, 0.0001]:
    print(f"*** EPSILON = {epsilon} ***")
    A, b, x = question_4_1(epsilon=epsilon)
    
    for method in  ["PLU", "QR"]:
        res = solve(A, b, method=method)
        print(f"\t{method} L1 error: {np.linalg.norm(x-res, ord=1)}")

    for method in  ["Jacobi", "GS", "SOR"]:
        if epsilon == 1 or epsilon == 0.1:
            omega = 1.5 # avoid overflow
        else:
            omega = None
        res = solve_iterative(A, b, method=method, verbose=True, tol=0.0001, omega=omega)
        print(f"\t{method} L1 error: {np.linalg.norm(x-res, ord=1)}")



