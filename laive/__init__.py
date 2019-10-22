from . factorization_LU import solve_LU, solve_PLU
from . factorization_Cholesky import solve_Cholesky, solve_LDL
from . factorization_QR import solve_QR
from . basic import solve_L, solve_U

from . iteration import solve_iterative
from . inversion import inverse

def solve(A, b, method="PLU"):
    if method == "PLU":
        return solve_PLU(A, b)
    elif method == "LDL":
        return solve_LDL(A, b)
    elif method == "QR":
        return solve_QR(A, b)
    elif method == "Cholesky":
        return solve_Cholesky(A, b)
    elif method == "LU":
        return solve_LU(A, b)
