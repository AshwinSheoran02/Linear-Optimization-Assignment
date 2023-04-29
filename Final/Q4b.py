from Q4a import integralPointSolver , findObjectiveCost
import numpy as np

if __name__ == "__main__":
    A = np.loadtxt("A_Primal.txt")
    b = np.loadtxt("B_Primal.txt")
    c = np.loadtxt("C_Primal.txt")

    primalSol = integralPointSolver(A, b, c)

    print("The solution vector is :")
    print(primalSol)

    print("The objective value is :")
    print(findObjectiveCost(c, primalSol))