import numpy as np
from numpy import linalg as LA
import math
import matplotlib.pyplot as plt


def richardson(epsilon, q, r, f, x_0, h, x_n, alpha, beta):

    rez = []
    N = []
    v1 = grid_method(q, r, f, x_0, h, x_n, alpha, beta)
    m = 1
    while m > epsilon:
        N.append((x_n - x_0) / h)
        h /= 2
        v2 = grid_method(q, r, f, x_0, h, x_n, alpha, beta)
        delta0 = (v2[0] - v1[0])/(pow(2, 1) - 1)
        m = -1
        for j in range(int(np.ceil((x_n - x_0) / h))):
            try:
                delta2 = (v2[2*j] - v1[j])/(pow(2, 1) - 1)
                delta1 = (delta0 + delta2) / 2
                m = max(abs(delta0),
                        abs(delta1),
                        abs(delta2),
                        m
                        )
                delta0 = delta2
            except IndexError:
                pass
        rez.append(m)
        v1 = v2
    print("количество итераций {0}".format(len(N)))
    return [rez, N[-1], v1]


def grid_method(q, r, f, x_0, h, x_n, alpha, beta):

    x = np.arange(x_0, x_n, h)
    n = len(x)
    u_0 = [alpha[0] + (3*alpha[1])/(2*h),
           -(2*alpha[1])/h,
           alpha[1]/(2*h)]
    u_n = [beta[0] + (3 * beta[1]) / (2 * h),
           -(2 * beta[1]) / h,
           beta[1] / (2 * h)]

    a = np.zeros([n, n])
    a[0, 0:3] = u_0
    a[n-1, n-3: n+1] = u_n

    b = np.zeros(n)
    b[0] = alpha[2]
    b[1: n - 1] = [f(x) for x in x[1: n - 1]]
    b[n - 1] = beta[2]

    Q = [q(x) for x in x[1: n - 1]]
    R = [r(x) for x in x[1: n - 1]]

    for i in range(1, n - 1):
        a[i, i - 1] = 1 / h**2 - Q[i - 1] / (2 * h)
        a[i, i] = -(2 / h**2 + R[i - 1])
        a[i, i + 1] = 1 / h**2 + Q[i - 1] / (2 * h)

    return LA.solve(a, b)


if __name__ == "__main__":
    epsilon = 1e-2

    def q1(x):
        return - (1 + x / 2) * (x - 3)

    def r1(x):
        return (x - 3) * math.exp(x/2)

    def f1(x):
        return - (2 - x) * (x - 3)

    alpha1 = [1, 0, 0]
    beta1 = [1, 0, 0]

    rich = richardson(epsilon, q1, r1, f1, -1, 0.2, 1, alpha1, beta1)
    print("погрешности 1: {0}".format(rich[0]))
    fig, ax = plt.subplots()
    ax.plot(np.linspace(1, -1, 2 * int(rich[1])), rich[2])
    plt.show()

    def q2(x):
        return -math.cos(x) / (1 + x)

    def r2(x):
        return 2 - x

    def f2(x):
        return x + 1

    alpha2 = [0.2, 1, -0.8]
    beta2 = [0.9, 1, -0.1]
    epsilon2 = 1e-3
    rich = richardson(epsilon2, q2, r2, f2, 0, 0.1, 1, alpha2, beta2)
    print("погрешности 2: {0}".format(rich[0]))
    fig, ax = plt.subplots()
    ax.plot(np.linspace(1, -1, 2 * int(rich[1])), rich[2])
    plt.show()