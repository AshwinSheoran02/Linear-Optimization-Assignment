import numpy as np


def revisedSimplexMethod(c, A, b):
    no_of_equations, no_of_variables = A.shape
    N = []
    B = []
    for i in range(no_of_variables - no_of_equations):
        N.append(no_of_variables - i - 1)
    for i in range(1, no_of_variables + 1):
        if i - 1 not in N:
            B.append(i - 1)

    A_B = A[:, B]
    x = np.linalg.solve(A_B, b)
    for i in range(len(N)):
        x = np.append(x, [0])
    x = np.reshape(x, (len(x), 1))
    while True:
        A_B = A[:, B]
        c_B = c[B]
        y_T = np.linalg.solve(A_B.T, c_B)
        k = -1
        for i in N:
            A_i = A[:, i]
            A_i = np.reshape(A_i, (len(A_i), 1))
            mul = np.matmul(y_T.T, A_i)
            if mul > c[i]:
                k = i
        if k == -1:
            break
        d_b = np.linalg.solve(A_B, -A[:, k])
        d = np.zeros((len(N) + len(B), 1))
        for i in range(len(B)):
            d[B[i]] = d_b[i]
        d[k] = 1
        possible_candidates = []
        min_val = 1e9
        for i in range(len(d_b)):
            if d_b[i] < 0:
                r = x[B[i]] / -d_b[i]
                min_val = min(r, min_val)
                possible_candidates.append([i, r])

        if len(possible_candidates) == 0:
            print("UnBounded")
            return -1

        leaving_var = -1
        for i in possible_candidates:
            if i[1] == min_val:
                leaving_var = B[i[0]]
                break

        B.append(k)
        N.append(leaving_var)
        B.remove(leaving_var)
        N.remove(k)
        x = x + min_val * d
    return x


if __name__ == '__main__':
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
    x = revisedSimplexMethod(c, A, b).T
    print("The optimal solution is: ",)
    print(x)

    print("The optimal value is: ",)
    print(np.dot(x, c))
