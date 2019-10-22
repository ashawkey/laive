import numpy as np

from . basic import solve_L, solve_U

def factorize_Cholesky(A):
    # Cholesky decomposition, return L
    n, n = A.shape
    L = np.copy(A)
    for i in range(n):
        L[i,i] = np.sqrt(np.abs(L[i,i]))
        L[i+1:n, i] /= L[i,i]
        for j in range(i+1, n):
            L[j:n, j] -= L[j:n, i] * L[j, i]
    return L

def solve_Cholesky(A, b):
    L = factorize_Cholesky(A)
    return solve_U(L.T, solve_L(L, b))

def factorize_LDL(A):
    # LDL' decomposition, LD compressed in one matrix.
    n, n = A.shape
    L = np.copy(A)
    v = np.zeros(n)
    for i in range(n):
        for j in range(i):
            v[j] = L[i, j] * L[j, j]
        L[i, i] -= L[i, 0:i] @ v[0:i]
        L[i+1:n, i] -= L[i+1:n, 0:i] @ v[0:i]
        L[i+1:n, i] /= L[i, i]
    return L

def solve_LDL(A, b):
    L =  factorize_LDL(A)
    dd = np.diag(L) ** -1
    return solve_U(L.T, dd * solve_L(L, b, use_LU=True), use_LU=True)