from Q1 import revisedSimplexMethod
import numpy as np

A=np.array([[1,1,0,0,0,0,0,0,-1,0,0,0],
            [0,1,1,0,0,0,0,0,0,-1,0,0],
            [0,1,0,1,0,0,0,0,0,0,-1,0],
            [0,0,1,1,0,0,0,0,0,0,0,-1],
            [1,0,0,0,1,0,0,0,0,0,0,0],
            [0,1,0,0,0,1,0,0,0,0,0,0],
            [0,0,1,0,0,0,1,0,0,0,0,0],
            [0,0,0,1,0,0,0,1,0,0,0,0]])
b=np.array([1,1,1,1,1,1,1,1])
c=np.array([1,1,1,1,0,0,0,0,0,0,0,0])


x = revisedSimplexMethod(c, A, b)

if(x.all()==-1):
    print("The problem is unbounded")
else:
    print("The Optimal Solution is: ",np.dot(c,x))