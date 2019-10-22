import numpy as np

from laive import solve, solve_iterative
from laive.questions import question_4_1


for epsilon in [1, 0.1, 0.01, 0.0001]:
    print(f"*** EPSILON = {epsilon} ***")
    A, b, x = question_4_1(n=10, epsilon=epsilon)
    #print("x:", x)
        
    for method in  ["PLU", "QR"]:
        res = solve(A, b, method=method)
        #print(res)
        print(f"\t{method} L1 error: {np.linalg.norm(x-res, ord=1)}")

    for method in  ["Jacobi", "GS", "SOR"]:
        res = solve_iterative(A, b, method=method, verbose=True, tol=0.0001, omega=None)
        #print(res)
        print(f"\t{method} L1 error: {np.linalg.norm(x-res, ord=1)}")
    