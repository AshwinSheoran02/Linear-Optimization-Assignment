from Q4a import integralPointSolver , findObjectiveCost

import numpy as np

if __name__ == "__main__":
    A = np.loadtxt("A_Dual.txt")
    b = np.loadtxt("B_Dual.txt")
    c = np.loadtxt("C_Dual.txt")

    dualSol = integralPointSolver(A, b, c)

    print("The solution vector is :")
    print(dualSol)

    print("The objective value is :")
    print(findObjectiveCost(c, dualSol))