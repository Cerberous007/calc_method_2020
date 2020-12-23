import numpy as np
import scipy as sp
import scipy.linalg as sp_linalg
import numpy.linalg as lin
import matplotlib.pyplot as plt

# test 1
def a(x, t):
    return 1

def f(x, t):
    return 2*t - 1

def phi(x):
    return x**2

def alpha1(t):
    return 1

def alpha2(t):
    return 0

def alpha(t):
    return t**2

def beta1(t):
    return 1

def beta2(t):
    return 0

def beta(t):
    return 1+ t**2

'''# test 2
def a(x, t):
    return 1

def f(x, t):
    return 2*np.cos(2*x)*(np.cos(2*t + np.pi/2) + 2*np.sin(2*t + np.pi/2))

def phi(x):
    return np.cos(2*x)

def alpha1(t):
    return 1

def alpha2(t):
    return 0

def alpha(t):
    return np.sin(2*t + np.pi/2)

def beta1(t):
    return 1

def beta2(t):
    return 0

def beta(t):
    return np.sin(2*t + np.pi/2)*np.cos(2)'''


def Tridiag_alg(n, a, c, b, g, x):
    temp = 0
    for i in range(1, n):
        m = a[i] / c[i - 1]
        c[i] = c[i] - m * b[i - 1]
        g[i] = g[i] - m * g[i - 1]

    x[n - 1] = g[n - 1] / c[n - 1]

    for i in range(n - 2, -1, -1):
        x[i] = (g[i] - b[i] * x[i + 1]) / c[i]
    return x


def Implicit_scheme(t, l, m, n):
    tau = t / m
    h = l / n
    u = np.zeros((m + 1, n + 1))
    x = np.zeros(n + 1)
    A = np.zeros(n + 1)
    B = np.zeros(n + 1)
    C = np.zeros(n + 1)
    G = np.zeros(n + 1)

    for i in range(n + 1):
        u[i][0] = phi(i * h)

    for k in range(1, m + 1):
        A[0] = 0
        B[0] = alpha1(tau * k) + alpha2(tau * k) / h
        C[0] = -alpha2(tau * k) / h
        G[0] = alpha(tau * k)

        A[n] = -beta2(tau * k) / h
        B[n] = beta1(tau * k) + beta2(tau * k) / h
        C[n] = 0
        G[n] = beta(tau * k)

        for i in range(1, n):
            A[i] = a(i * h, tau * k) / h ** 2
            B[i] = -2 * a(i * h, tau * k) / (h ** 2) - 1 / (tau)
            C[i] = a(i * h, tau * k) / (h ** 2)
            G[i] = -(u[i, k - 1]) / (tau) - f(i * h, tau * k)
        x = np.zeros(n + 1)
        x = Tridiag_alg(n + 1, A, B, C, G, x)

        for j in range(n + 1):
            u[j][k] = x[j]
    print(u)


def Explicit_scheme(t, l, m, n):
    tau = t / m
    h = l / n
    u = np.zeros((m + 1, n + 1))
    for i in range(n + 1):
        u[i][0] = phi(i * h)

    for k in range(1, m + 1):
        for i in range(1, n):
            u[i, k] = u[i, k - 1] + \
                      tau * (a(i * h, (k - 1) * tau) * (u[i + 1, k - 1] - 2 * u[i, k - 1] + u[i - 1, k - 1]) / \
                             h ** 2 + f(i * h, (k - 1) * tau))
        u[0, k] = (alpha(k * tau) + alpha2(k * tau) * \
                   (-3 * u[0, k] + 4 * u[1, k] - 2 * u[2, k]) / (2 * h)) / alpha1(k * tau)
        u[n, k] = (beta(k * tau) - beta2(k * tau) * (3 * u[n, k] - 4 * \
                                                     u[n - 1, k] + u[n - 2, k]) / (2 * h)) / beta1(k * tau)

    print(u)

t, l, m, n = 1, 1, 50, 50
Implicit_scheme(t, l, m, n)
Explicit_scheme(t, l, m, n);
