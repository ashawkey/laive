import numpy as np
from scipy.linalg import cholesky, solve
from . basic import solveL, solveU

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
    return solveU(L.T, solveL(L, b))

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
    return solveU(L.T, dd * solveL(L, b, use_LU=True), use_LU=True)

if __name__ == "__main__":
    # (1)
    l = 100
    A = np.zeros((l, l))
    for i in range(l):
        A[i, i] = 10
        if i != 0: A[i, i-1] = 1
        if i != l-1: A[i, i+1] = 1
    
    cholesky_error = []
    ldl_error = []
    for seed in range(100):
        np.random.seed(seed)
        b = np.random.rand(100)

        res_scipy = solve(A, b) # use scipy as the ground truth
        res_cholesky = solve_Cholesky(A, b)
        res_LDL = solve_LDL(A, b)

        #print(res_cholesky)
        cholesky_error.append(np.linalg.norm(res_scipy - res_cholesky, 1))
        #print(res_LDL)
        ldl_error.append(np.linalg.norm(res_scipy - res_LDL, 1))

    print("Cholesky L1 error: ", np.mean(cholesky_error))
    print("LDL L1 error: ", np.mean(ldl_error))

    # (2)
    l = 40
    A = np.zeros((l, l))
    b = np.zeros(l)
    for i in range(l):
        for j in range(l):
            A[i, j] = 1 / (i + j + 1);
            b[i] += A[i, j]
    
    res_truth = np.ones(l)
    #res_scipy = solve(A, b)
    res_cholesky = solve_Cholesky(A, b)
    res_LDL = solve_LDL(A, b)

    print(A)
    print(b)

    print(res_cholesky)
    print("Cholesky L1 error: ", np.linalg.norm(res_truth - res_cholesky, 1))
    print(res_LDL)
    print("LDL L1 error: ", np.linalg.norm(res_truth - res_LDL, 1))