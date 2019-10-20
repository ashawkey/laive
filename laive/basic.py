import numpy as np

def get_D(A):
    D = np.diag(np.diag(A))
    return D

def get_L(A):
    m, n = A.shape
    L = np.zeros_like(A)
    for i in range(m):
        for j in range(n):
            if j < i:
                L[i, j] = A[i, j]
    return L

def get_U(A):
    m, n = A.shape
    U = np.zeros_like(A)
    for i in range(m):
        for j in range(n):
            if j > i:
                U[i, j] = A[i, j]
    return U

def get_DLU(A):
    m, n = A.shape
    D = np.zeros_like(A)
    L = np.zeros_like(A)
    U = np.zeros_like(A)
    for i in range(m):
        for j in range(n):
            if j == i: D[i, j] = A[i, j]
            elif j > i: U[i, j] = A[i, j]
            else: L[i, j] = A[i, j]
    return D, L, U

def solveL(L, b, use_LU=False):
    # solve Lx = b, assume L is lower-triangular
    # use_LU: set L[i,i] to 1 if true
    n, n = L.shape
    x = np.copy(b)
    for i in range(n):
        if not use_LU:
            x[i] /= L[i, i]
        if i != n - 1:
            x[i+1:n] -= x[i] * L[i+1:n, i]
    return x

def solveU(U, b, use_LU=False):
    # solve Ux = b, assume U is upper-triangular
    # use_LU: set U[i,i] to 1 if true
    n, n = U.shape
    x = np.copy(b)
    for i in range(n-1, -1, -1):
        if not use_LU:
            x[i] /= U[i, i]
        if i != 0:
            x[0:i] -= x[i] * U[0:i, i]
    return x