import numpy as np
from numpy.linalg import norm
from numpy import linalg
from random import normalvariate
from math import sqrt

def random_unit_vector(n):
    unnormalized = [normalvariate(0, 1) for _ in range(n)]
    norm = sqrt(sum(x * x for x in unnormalized))
    return [x / norm for x in unnormalized]

def svd_1d(A, eps=1e-10):
    n, m = A.shape
    x = random_unit_vector(m)
    lastV = None
    currentV = x
    B = np.dot(A.T, A)

    iterations = 0
    while True:
        iterations += 1
        lastV = currentV
        currentV = np.dot(B, lastV)
        currentV = currentV / np.linalg.norm(currentV)

        if abs(np.dot(currentV, lastV)) > 1 - eps:
            print("converged in {} iterations!".format(iterations))
            return currentV


def power_method_svd(A):
    n, m = A.shape
    svdSoFar = []

    for i in range(m):
        matrixFor1D = A.copy()

        for singularValue, u, v in svdSoFar[:i]:
            matrixFor1D = matrixFor1D - singularValue * np.outer(u, v)

        v = svd_1d(matrixFor1D)
        u_unnormalized = np.dot(A, v)
        sigma = norm(u_unnormalized)
        u = u_unnormalized / sigma

        svdSoFar.append((sigma, u, v))

    singularValues, us, vs = [np.array(x) for x in zip(*svdSoFar)]

    return us.T, singularValues, vs

