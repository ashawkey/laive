from . factorization_LU import solveLU, solvePLU
from . factorization_Cholesky import solve_Cholesky, solve_LDL
from . factorization_QR import solveQR
from . basic import solveL, solveU

from . iteration import solve_iterative
from . inversion import inverse

def solve(A, b, method="PLU"):
    if method == "PLU":
        return solvePLU(A, b)
    elif method == "LDL":
        return solve_LDL(A, b)
    elif method == "QR":
        return solveQR(A, b)
    elif method == "Cholesky":
        return solve_Cholesky(A, b)
    elif method == "LU":
        return solveLU(A, b)
