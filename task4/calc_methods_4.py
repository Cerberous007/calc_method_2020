import numpy as np
import numpy.linalg as lin

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


# Метод якоби с циклияным выбором элемента
def jacobi_cycl(a, eps):

    n = np.shape(a)[0]
    m = max_in_mattrix(np.abs(np.triu(a, 1)))
    count = 0

    while m[0] > eps:
        for i in range(n):
            for j in range(i+1, n):
                T = np.eye(n)
                if a[i][i] != a[j][j]:
                    phi = np.arctan(-2 * a[i][j] / (a[j][j] - a[i][i])) / 2
                else:
                    phi = np.pi / 4
                c = np.cos(phi)
                s = np.sin(phi)

                T[i][i] = c
                T[j][j] = c
                T[i][j] = -s
                T[j][i] = s

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

n = 6
a = hilb(n)
m = jacobi_max(a, 1e-12)
print("$\lambda_i$ Якоби с max элементом:\n\n{0} \n\n iterations: {1}\n\n".format(
    sorted(m[0]), m[1]))
c = jacobi_cycl(a, 1e-12)
print("$\lambda_i$ Якоби с циклическим выбором:\n\n{0} \n\n iterations: {1}\n\n".format(
    sorted(c[0]), c[1]))
print("\lambdas with numpy:\n\n{0}".format(sorted(lin.eig(a)[0])))