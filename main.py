from math import ceil

import numpy as np
import matplotlib.pyplot as plt

l = 10 * 10 ** (-6)  # 10mkm
L = 10 * 10 ** (-6)  # 10mkm

lbd = 2 * 10 ** (-6)  # 2mkm
n = 1
k = 2 * np.pi / lbd
M = 1j / (2 * k * n)

count = 500


def psi(x):
    return 5 * np.exp(-((2 * x - l) / (0.2 * l)) ** 2)


def implicit_solution(x, z):
    I, K = len(x) - 1, len(z) - 1
    h_x = x[1] - x[0]
    h_z = z[1] - z[0]

    gamma = - M * h_z / (h_x ** 2)

    u = np.zeros((K + 1, I + 1), dtype=complex)

    for i in range(0, I):
        u[0][i] = psi(x[i])

    for k in range(1, K + 1):
        p, q = np.zeros(I, dtype=complex), np.zeros(I, dtype=complex)
        p[0] = 0
        q[0] = 0

        for i in range(1, I):
            temp = (1 + (1.5 - 0.75 * p[i - 1]) * gamma)
            v = (1 - 0.5 * gamma) * u[k - 1, i] + 0.25 * gamma * (u[k - 1, i + 1] + u[k - 1, i - 1])
            p[i] = 0.75 * gamma / temp
            q[i] = (0.75 * gamma * q[i - 1] + v) / temp

        u[k, I] = 0
        for i in range(I - 1, -1, -1):
            u[k, i] = p[i] * u[k, i + 1] + q[i]

    return u


def plot(result, arr, arr2, values, label):
    h = arr[1] - arr[0]
    for value in values:
        j = ceil(value / h)
        if label == 'x':
            plt.plot(arr2, np.abs(result[:, j]), label=label + ' = ' + str(value))
            plt.xlabel('z')
        else:
            plt.plot(arr2, np.abs(result[j, :]), label=label + ' = ' + str(value))
            plt.xlabel('x')

    plt.ylabel("U(x, z)")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    x_array = np.linspace(0, l, count)
    z_array = np.linspace(0, L, count)

    u = implicit_solution(x_array, z_array)

    x_values = [0.01 * 10 ** (-6),
                3. * 10 ** (-6),
                6. * 10 ** (-6),
                10. * 10 ** (-6)]
    z_values = [0.01 * 10 ** (-6),
                3.334 * 10 ** (-6),
                6.667 * 10 ** (-6),
                10. * 10 ** (-6)]
    plot(u, z_array, x_array, z_values, 'z')
    plot(u, x_array, z_array, x_values, 'x')
