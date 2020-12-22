import numpy as np
import numpy.linalg as lin

def power_method(a, eps):
    x = np.random.random(np.shape(a)[0])
    X = np.dot(a, x)
    l = lin.norm(X[0])/lin.norm(x[0])
    count = 1
    delta = 1
    while delta > eps:
        delta = l
        x = X / lin.norm(np.dot(a, x))
        X = np.dot(a, x)
        l = lin.norm(X[0]) / lin.norm(x[0])
        delta = abs(delta - l)
        count += 1
    return l, count

def scalar_method(a, eps):
    x = np.random.random(np.shape(a)[0])
    X = np.dot(a, x) / lin.norm(np.dot(a, x))
    y = x
    Y = np.dot(np.transpose(a), x) / lin.norm(np.dot(np.transpose(a), y))
    l = np.dot(np.dot(a, X), np.dot(np.transpose(a), Y)) / np.dot(X, np.dot(np.transpose(a), Y))
    count = 1
    delta = 1
    while delta > eps:
        delta = l
        x = X
        X = np.dot(a, x) / lin.norm(np.dot(a, x))
        y = Y
        Y = np.dot(np.transpose(a), x) / lin.norm(np.dot(np.transpose(a), y))
        l = np.dot(np.dot(a, X), np.dot(np.transpose(a), Y)) / np.dot(X, np.dot(np.transpose(a), Y))
        delta = abs(delta - l)
        count += 1
    return l, count

def max_in_mattrix(a):
    n = np.shape(a)[0]
    m = a[0][0]
    max_i = 0
    max_j = 0
    for j in range(n):
        for i in range(n):
            if i != j and m < a[i][j]:
                m = a[i][j]
                max_i = i
                max_j = j
    return m, max_i, max_j

# Якоби с выбором максимального элемента
def jacobi_max(a, eps):

    n = np.shape(a)[0]
    m = max_in_mattrix(np.abs(np.triu(a, 1)))
    count = 0

    while m[0] > eps:
        T = np.eye(n)
        if a[m[1]][m[1]] != a[m[2]][m[2]]:
            phi = np.arctan(-2 * a[m[1]][m[2]] / (a[m[2]][m[2]] - a[m[1]][m[1]])) / 2
        else:
            phi = np.pi / 4
        c = np.cos(phi)
        s = np.sin(phi)

        T[m[1]][m[1]] = c
        T[m[2]][m[2]] = c
        T[m[1]][m[2]] = -s
        T[m[2]][m[1]] = s

        a = np.dot(np.dot(np.transpose(T), a), T)
        m = max_in_mattrix(np.abs(np.triu(a, 1)))
        count += 1
    return np.diag(a), count

# Создаем матрицу Гильберта
def hilb(n):
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            H[i][j] = 1/(i+j+1)
    return H

n = 10
eps = 1e-12
a = hilb(n)
p = power_method(a, eps)
s = scalar_method(a, eps)
print("max $\lambda$ степенной метод: {0};\n iterations: {1}\n".format(p[0], p[1]))
print("max $\lambda$ метод скалярных: {0}; \n iterations: {1}\n".format(s[0], s[1]))
j = jacobi_max(a, eps)
print("max $\lambda$ метод вращений якоби: {0};\n iterations: {1}\n".format(max(j[0]), j[1]))
print("max $\lambda$ numpy: {0};\n".format(max(lin.eig(a)[0])))