import numpy as np
import sympy

eps = 1e-4

def prepareA(A,b,c):
    m, n = A.shape
    if(m != b.shape[0]):
        print("A and b are not compatible")
        return
    if(n != c.shape[0]):
        print("A and c are not compatible")
        return

    rankA = np.linalg.matrix_rank(A)
    if(rankA != min(m, n)):
        _, rrefPivot = sympy.Matrix(A).T.rref()
        A = A[list(rrefPivot)]

    return A

def integralPointSolver(A, b, c):
    m, n = A.shape
    modifiedA = prepareA(A, b,c)
    if(modifiedA.all() == None):
        return
    A = modifiedA

    slackSol = np.ones(n)
    primalSol = np.ones(n)
    dualSol = np.ones(m)

    xDotS = np.dot(primalSol, slackSol)
    while(True):
        muK = xDotS/n
        sigmaK = 0.4

        Anew = np.zeros((m+2*n, m+2*n))
        Anew[0:m, 0:n] = np.copy(A)
        Anew[m:m+n, n:m+n] = np.copy(A.T)
        Anew[m:m+n, m+n:m+2*n] = np.eye(n)
        Anew[m+n:m+2*n, 0:n] = np.copy(np.diag(slackSol))
        Anew[m+n:m+2*n, n+m:m+2*n] = np.copy(np.diag(primalSol))

        bnew = np.zeros(m+2*n)
        bnew[0:m] = np.copy(b-np.dot(A, primalSol))
        bnew[m:m+n] = np.copy(c-np.dot(A.T, dualSol)-slackSol)
        bnew[m+n:m+2*n] = np.copy(muK*sigmaK*np.ones(n)-np.dot(
            np.dot(np.diag(slackSol), np.diag(primalSol)), np.ones(n)))

        delta = np.linalg.solve(Anew, bnew)
        ds = delta[n+m:m+2*n]
        dx = delta[0:n]
        dl = delta[n:n+m]

        alphaMax = min(np.where(ds < 0, -slackSol/ds, np.inf).min(), np.where(dx < 0, -primalSol/dx, np.inf).min())


        etaK = 0.98
        alphaK = min(etaK*alphaMax, 1)

        slackSol = slackSol+alphaK*ds
        primalSol = primalSol+alphaK*dx
        dualSol = dualSol+alphaK*dl

        xDotS = np.dot(primalSol, slackSol)
        if(xDotS < eps):
            break

    return primalSol, dualSol


def findObjectiveCost(cost, solution):
    return np.dot(cost, solution)


if __name__ == "__main__":
    choice = int(input("Enter 1 to enter the data manually or 2 to use the default data : "))
    if(choice == 1):
        rowA = int(input("Enter the number of rows in A : "))
        colA = int(input("Enter the number of columns in A : "))
        A = np.zeros((rowA, colA))
        print("Enter the elements of A : ")
        for i in range(rowA):
            for j in range(colA):
                A[i][j] = int(input())

        rowB = int(input("Enter the number of rows in b : "))
        b = np.zeros(rowB)
        print("Enter the elements of b : ")
        for i in range(rowB):
            b[i] = int(input())

        col = int(input("Enter the number of columns in c : "))
        c = np.zeros(col)
        print("Enter the elements of c : ")
        for i in range(colA):
            c[i] = int(input())

    else:
        A=np.array([[4,8,6,-1,0],[3,6,12,0,-1]])
        b=np.array([64,96])
        c=np.array([12,8,15,0,0])

    if(integralPointSolver(A, b, c) == None):
        print("The problem is infeasible")
        exit()

    primalSol, dualSol = integralPointSolver(A, b, c)

    print("The solution vector is :")
    print(primalSol)

    print("The objective value is :")
    print(findObjectiveCost(c, primalSol))