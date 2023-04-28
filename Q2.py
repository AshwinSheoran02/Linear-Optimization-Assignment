from revisedSimplex import revisedSimplexMethod
import numpy as np

A=np.array([
    [1,0,0,0,0,0,0,0,1,0,0,0],
    [1,1,0,1,0,0,0,0,0,1,0,0],
    [0,1,1,0,0,0,0,0,0,0,1,0],
    [0,0,1,1,0,0,0,0,0,0,0,1],
    [1,0,0,0,1,0,0,0,0,0,0,0],
    [0,1,0,0,0,1,0,0,0,0,0,0],
    [0,0,1,0,0,0,1,0,0,0,0,0],
    [0,0,0,1,0,0,0,1,0,0,0,0]
])

# cost vector is multiplied by -1 to convert the maximisation problem to a minimisation problem
c=np.array([-1,-1,-1,-1,0,0,0,0,0,0,0,0])
b=np.array([1,1,1,1,1,1,1,1])


# We convert the maximisation problem to a minimisation problem by multiplying the objective function by -1
# so the optimal solution is -1*optimal solution of the minimisation problem
x = revisedSimplexMethod(c, A, b)

if(x.all()==-1):
    print("The problem is unbounded")
else:
    print("The Optimal Solution is: ",-np.dot(c,x))


