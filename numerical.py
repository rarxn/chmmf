import numpy as np
from math import ceil
import matplotlib.pyplot as plt


def psi(x, l):
    return 5 * np.exp(-((2 * x - l) / (0.2 * l)) ** 2)


def implicit_solution_nikitin(M, l, x, z):
    I, K = len(x) - 1, len(z) - 1
    h_x = x[1] - x[0]
    h_z = z[1] - z[0]

    gamma = - M * h_z / (h_x ** 2)

    u = np.zeros((K + 1, I + 1), dtype=complex)

    for i in range(0, I):
        u[0][i] = psi(x[i], l)

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


def implicit_solution_yashakin(M, l, x, z):
    I, K = len(x) - 1, len(z) - 1
    h_x = x[1] - x[0]
    h_z = z[1] - z[0]

    gamma = - M * h_z / (h_x ** 2)

    u = np.zeros((K + 1, I + 1), dtype=complex)

    for i in range(0, I):
        u[0][i] = psi(x[i], l)

    for k in range(1, K + 1):
        p, q = np.zeros(I, dtype=complex), np.zeros(I, dtype=complex)
        p[0] = 0
        q[0] = 0

        for i in range(1, I):
            temp = (1 + (1 - 0.5 * p[i - 1]) * gamma)
            v = (1 - gamma) * u[k-1, i] + 0.5 * gamma * (u[k-1, i + 1] + u[k-1, i - 1])
            p[i] = 0.5 * gamma / temp
            q[i] = (0.5 * gamma * q[i - 1] + v) / temp

        u[k, I] = 0
        for i in range(I - 1, -1, -1):
            u[k, i] = p[i] * u[k, i + 1] + q[i]

    return u


def implicit_solution_kasparov(M, l, x, z):
    I, K = len(x) - 1, len(z) - 1
    h_x = x[1] - x[0]
    h_z = z[1] - z[0]

    gamma = - M * h_z / (h_x ** 2)

    u = np.zeros((K + 1, I + 1), dtype=complex)

    for i in range(0, I):
        u[0][i] = psi(x[i], l)

    for k in range(1, K + 1):
        p, q = np.zeros(I, dtype=complex), np.zeros(I, dtype=complex)
        p[0] = 0
        q[0] = 0

        for i in range(1, I):  # при I - 1 сходимость хуже
            delitel = (1 + (2 - p[i - 1]) * gamma)
            p[i] = gamma / delitel
            q[i] = (gamma * q[i - 1] + u[k - 1, i]) / delitel

        u[k, I] = 0
        for i in range(I - 1, 0, -1):  # начало, стоп, инкремент
            u[k, i] = p[i] * u[k, i + 1] + q[i]

    return u


def get_sol(u, arr, values, fixed):
    h = arr[1] - arr[0]
    res = []
    for value in values:
        j = ceil(value / h)
        if fixed == 'x':
            res.append(u[:, j])
        else:
            res.append(u[j, :])
    return res


def get_numerical(M, l, L, x_arr, z_arr, x_val, z_val, var):
    if var == 1:
        u = implicit_solution_nikitin(M, l, x_arr, z_arr)
    if var == 2:
        u = implicit_solution_yashakin(M, l, x_arr, z_arr)
    if var == 3:
        u = implicit_solution_kasparov(M, l, x_arr, z_arr)
    return get_sol(u, z_arr, z_val, 'z'), get_sol(u, x_arr, x_val, 'x')


def main():
    l = 10.  # 10mkm
    L = 10.  # 10mkm

    lbd = 2.  # 2mkm
    n = 1
    k = 2 * np.pi / lbd
    M = 1j / (2 * k * n)

    count = 500

    x_array = np.linspace(0, l, count)
    z_array = np.linspace(0, L, count)

    x_values = [0.01, 3., 6., 10.]
    z_values = [0.01, L / 3, 2 * L / 3, L]

    x, z = get_numerical(M, l, L, x_array, z_array, x_values, z_values)

    [plt.plot(x_array, np.abs(_x), label=f'z= {val:.4} мкм') for _x, val in zip(x, z_values)]
    plt.xlabel("x, мкм")
    plt.ylabel("U(x, z)")
    plt.grid()
    plt.legend(loc='upper right')
    plt.show()

    [plt.plot(z_array, np.abs(_z), label=f'x= {val:.4} мкм') for _z, val in zip(z, x_values)]
    plt.xlabel("z, мкм")
    plt.ylabel("U(x, z)")
    plt.grid()
    plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    main()
