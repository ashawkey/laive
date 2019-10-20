import scipy
import numpy as np

from . factorization_Cholesky import solve_LDL
from . basic import solveL, solveU

#eps = np.finfo(np.float64).eps

def transform_Householder(x):
    # H @ x = [1, 0, 0, ...]
    # H = I - beta @ v @ v.T
    n = x.shape[0]
    v = np.copy(x)
    v = v / np.max(np.abs(v))
    sigma = v[1:].T @ v[1:]
    
    if sigma == 0:
        beta = 0
    else:
        alpha = np.sqrt(v[0]**2 + sigma)
        if v[0] <= 0:
            v[0] = v[0] - alpha
        else:
            v[0] = - sigma / (v[0] + alpha)
        beta = 2 * v[0]**2 / (sigma + v[0]**2)
        v = v / v[0]
    return beta, v

def retrieve_H(beta, v, length=None):
    n = v.shape[0]
    if length is None:
        length = n
    H = np.eye(length)
    v = np.hstack([[0]*(length-n), v])
    H -= beta * v.reshape(-1,1) @ v.reshape(1,-1)
    return H

def factorize_QR(A):
    # save QR in X, d
    m, n = A.shape
    d = np.zeros(n)
    X = np.copy(A)
    for j in range(n):
        if j < m:
            beta, v = transform_Householder(X[j:, j])
            X[j:, j:] = retrieve_H(beta, v) @ X[j:, j:]
            d[j] = beta
            X[j+1:, j] = v[1:]
    return X, d

def retrieve_QR(X, d):
    m, n = X.shape
    Q = np.eye(m)
    R = np.zeros_like(X)
    for j in range(n):
        beta = d[j]
        v = np.hstack([[1], X[j+1:, j]])
        H = retrieve_H(beta, v, m)
        Q = Q @ H
        R[j, j:] = X[j, j:]
    return Q, R

def solveQR(A, b):
    Q, R = retrieve_QR(*factorize_QR(A))
    #Q, R = scipy.linalg.qr(A)
    c = Q.T @ b
    return solveU(R, c)

def LS_regularized(A, b):
    C = A.T @ A
    d = A.T @ b
    return solve_LDL(C, d)

def LS_QR(A, b):
    C = A.T @ A
    d = A.T @ b
    return solveQR(C, d)


if __name__ == "__main__":
    ## (2)
    t = np.array([-1, -0.75, -0.5, 0, 0.25, 0.5, 0.75])
    y = np.array([1, 0.8125, 0.75, 1, 1.3125, 1.75, 2.3125])

    A = np.ones((7, 3))
    for i in range(7):
        A[i, 0] = t[i] * t[i]
        A[i, 1] = t[i]

    res_ldl = LS_regularized(A, y)
    res_qr = LS_QR(A, y)

    print(res_ldl)
    print("res_ldl error",  np.linalg.norm(A@res_ldl - y, ord=2))
    print(res_qr)
    print("res_qr error",  np.linalg.norm(A@res_qr - y, ord=2))
    

    



