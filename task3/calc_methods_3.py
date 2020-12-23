import numpy as np
from numpy import linalg as LA
from scipy.sparse import coo_matrix
import random


def simple_iter(a, b, epsilon):
    n = np.shape(a)[0]
    B = np.zeros([n, n])
    beta = [0]*n
    for i in range(n):
        for j in range(n):
            if i != j:
                B[i][j] = -a[i][j] / a[i][i]
        beta[i] = b[i]/a[i][i]

    if np.max(np.abs(LA.eig(B)[0])) < 1:
        x0 = beta
        x = np.dot(B, x0) + beta
        count = 1
        while LA.norm(x - x0) > epsilon:
            x0 = x
            x = np.dot(B, x0) + beta
            count += 1
        return x, count
    else:
        print(np.max(np.abs(LA.eig(B)[0])))
        print("Метод не сходится")


def zeidel(a, b, epsilon):
    n = np.shape(a)[0]
    B = np.zeros([n, n])
    E = np.eye(n)
    beta = [0]*n
    for i in range(n):
        for j in range(n):
            if i != j:
                B[i][j] = -a[i][j] / a[i][i]
        beta[i] = b[i]/a[i][i]

    Hr = np.triu(B)
    Hl = np.tril(B, -1)
    B = np.dot(LA.inv(E - Hl), Hr)
    beta = np.dot(LA.inv(E - Hl), beta)
    if np.max(np.abs(LA.eig(B)[0])) < 1:
        x0 = beta
        x = np.dot(B, x0) + beta
        count = 1
        while LA.norm(x - x0) > epsilon:
            x0 = x
            x = np.dot(B, x0) + beta
            count += 1
        return x, count
    else:
        print(np.max(np.abs(LA.eig(B)[0])))
        print("Метод не сходится")



print("первый тест")
test1 = {
    "a": np.array([[4.0, -1.0, 0.0, -1.0, 0.0],
                   [-1.0, 4.0, -1.0, 0.0, 0.0],
                   [0.0, -1.0, 4.0, 0.0, -1.0],
                   [-1.0, 0.0, 0.0, 4.0, -1.0],
                   [0.0, -1.0, 0.0, -1.0, 4.0]]),
    "b": np.array([0.0, 0.0, 0.0, 100.0, 100.0])
}
print("решение методом простой итерации")
print(simple_iter(test1["a"], test1["b"], 1e-10))
print("решение методом Зейделя")
print(zeidel(test1["a"], test1["b"], 1e-10))

print("второй тест")
# далее будет выводиться симметричная разряженная матрица 50 на 50
row = []
col = []
for i in range(50):
    rand_place1 = random.randint(0, 49)
    rand_place2 = random.randint(0, 49)
    row.append(rand_place1)
    col.append(rand_place2)
    row.append(rand_place2)
    col.append(rand_place1)

for i in range(50):
    row.append(i)
    col.append(i)

data = []
for i in range(100):
    if i < 50:
        r = random.uniform(0.1, 0.3)
        data.append(r)
        data.append(r)
    else:
        data.append(random.uniform(0.8, 0.9))

coo = coo_matrix((np.array(data), (np.array(row), np.array(col))), shape=(50, 50))
with open('test.txt', 'wb') as f:
    np.savetxt(f, np.column_stack(coo.toarray()), fmt='%1.10f')
print("решение методом простой итерации")
print(simple_iter(coo.toarray(), [1]*50, 1e-10))
print("решение методом Зейделя")
print(zeidel(coo.toarray(), [1]*50, 1e-10))