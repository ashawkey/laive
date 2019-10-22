import numpy as np

from . basic import solve_L, solve_U

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

def solve_LU(A, b):
    LU = factorize_LU(A)
    return solve_U(LU, solve_L(LU, b, use_LU=True))

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

def solve_PLU(A, b):
    P, LU = factorize_PLU(A)
    return solve_U(LU, solve_L(LU, P@b, use_LU=True))