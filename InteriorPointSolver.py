import numpy as np
import sympy

eps = 1e-4  # tolerance for stopping criterion

# function to prepare the input matrices for solving
def prepareA(A, b, c):
    m, n = A.shape
    rankA = np.linalg.matrix_rank(A)
    # if matrix A is not full rank, drop redundant rows using row-reduced echelon form
    if rankA != min(m, n):
        _, rrefPivot = sympy.Matrix(A).T.rref()
        A = A[list(rrefPivot)]

    return A

# function to solve the integral point linear program using interior point method
def integralPointSolver(A, b, c):
    m, n = A.shape
    modifiedA = prepareA(A, b, c)  # prepare the input matrices
    if modifiedA.all() == None:
        # if there is no solution, return None
        return None
    A = modifiedA

    # initialize the solution with all ones
    slackSol = np.ones(n)
    primalSol = np.ones(n)
    dualSol = np.ones(m)

    # calculate the initial dot product of primalSol and slackSol
    xDotS = np.dot(primalSol, slackSol)

    # iterate until the dot product of primalSol and slackSol is less than the tolerance value eps
    while True:
        # calculate the current value of the barrier parameter
        muK = xDotS / n

        # set the centering parameter
        sigmaK = 0.4

        # construct the augmented matrix for solving the system of linear equations
        Anew = np.zeros((m + 2*n, m + 2*n))
        Anew[0:m, 0:n] = np.copy(A)
        Anew[m:m+n, n:m+n] = np.copy(A.T)
        Anew[m:m+n, m+n:m+2*n] = np.eye(n)
        Anew[m+n:m+2*n, 0:n] = np.copy(np.diag(slackSol))
        Anew[m+n:m+2*n, n+m:m+2*n] = np.copy(np.diag(primalSol))

        # construct the right-hand side of the linear system
        bnew = np.zeros(m + 2*n)
        bnew[0:m] = np.copy(b - np.dot(A, primalSol))
        bnew[m:m+n] = np.copy(c - np.dot(A.T, dualSol) - slackSol)
        bnew[m+n:m+2*n] = np.copy(muK * sigmaK * np.ones(n) - np.dot(np.dot(np.diag(slackSol), np.diag(primalSol)), np.ones(n)))

        # solve the linear system to obtain the primal, dual, and slack variables
        delta = np.linalg.solve(Anew, bnew)
        ds = delta[n+m:m+2*n]
        dx = delta[0:n]
        dl = delta[n:n+m]

        # calculate the maximum step length for primal and slack variables
        alphaMax = min(np.where(ds < 0, -slackSol/ds, np.inf).min(), np.where(dx < 0, -primalSol/dx, np.inf).min())

        # set the step length reduction factor
        etaK = 0.98

        # calculate the step length
        alphaK = min(etaK * alphaMax, 1)

        # update the primal, dual, and slack variables using the step lengths
        slackSol = slackSol + alphaK * ds
        primalSol = primalSol+alphaK*dx
        dualSol = dualSol+alphaK*dl

        xDotS = np.dot(primalSol, slackSol)
        if(xDotS < eps):
            break

    return primalSol


def findObjectiveCost(cost, solution):
    return np.dot(cost, solution)


if __name__ == "__main__":
    A=np.array([[4,8,6,-1,0],[3,6,12,0,-1]])
    b=np.array([64,96])
    c=np.array([12,8,15,0,0])

    primalSol = integralPointSolver(A, b, c)

    print("The solution vector is :")
    print(primalSol)

    print("The objective value is :")
    print(findObjectiveCost(c, primalSol))