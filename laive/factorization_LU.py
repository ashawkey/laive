import numpy as np
from scipy.linalg import lu, solve
from . basic import solveL, solveU

def factorize_LU(A):
    # LU decomposition, LU compressed in one matrix
    m, n = A.shape
    assert m == n
    X = np.copy(A)
    for i in range(n-1):
        X[i+1:n, i] = X[i+1:n, i] / X[i, i]
        X[i+1:n, i+1:n] -= X[i+1:n, i][:,np.newaxis] @ X[i, i+1:n][np.newaxis,:]
    return X

def retrieve_LU(LU):
    m, n = LU.shape
    assert m == n
    L = np.zeros_like(LU)
    U = np.zeros_like(LU)
    for i in range(n):
        for j in range(n):
            if i == j:
                L[i, j] = 1
                U[i, j] = LU[i, j]
            elif i > j:
                L[i, j] = LU[i, j]
            else:
                U[i, j] = LU[i, j]
    return L, U

def solveLU(A, b):
    LU = factorize_LU(A)
    return solveU(LU, solveL(LU, b, use_LU=True))

def factorize_PLU(A):
    # PLU decomposition, LU compressed in one matrix
    n, n = A.shape
    X = np.copy(A)
    P = np.eye(n)
    for i in range(n-1):
        idx = np.argmax(np.abs(X[i:, i]))
        X[[i, i+idx]] = X[[i+idx, i]]
        P[[i, i+idx]] = P[[i+idx, i]]
        X[i+1:n, i] /= X[i, i]
        X[i+1:n, i+1:n] -= X[i+1:n, i][:,np.newaxis] @ X[i, i+1:n][np.newaxis,:]
    return P, X

def solvePLU(A, b):
    P, LU = factorize_PLU(A)
    return solveU(LU, solveL(LU, P@b, use_LU=True))

if __name__ == "__main__":
    l = 84
    A = np.zeros((l, l))
    b = np.zeros(l)

    for i in range(l):
        A[i, i] = 6
        if i != 0: A[i, i-1] = 8
        if i != l-1: A[i, i+1] = 1

        if i == 0: b[i] = 7
        elif i == l-1: b[i] = 14
        else: b[i] = 15
    
    #print(A)
    #print(b)
    
    #res_scipy = solve(A, b) # scipy solution
    A = A.astype(np.float64)
    b = b.astype(np.float64)

    res_true = np.ones(l)
    res_lu = solveLU(A, b)
    res_plu = solvePLU(A, b)

    #print(res_lu)
    print("LU L1 error:", np.linalg.norm(res_lu-res_true, ord=1))
    #print(res_plu)
    print("PLU L1 error:", np.linalg.norm(res_plu-res_true, ord=1))


