import numpy as np
from scipy import special


def n_min(eps):
    N = 1
    while (np.sqrt(np.pi) * 400 * np.exp(-(np.pi * N / 20) ** 2) / (2 * N * np.pi ** 2)) > eps:
        N = N + 1
    return N


def n_exp(x_arr, z, N, eps):
    i = N
    while all([abs(U(x, z, i) - U(x, z, i - 1)) < eps for x in x_arr]):
        i -= 1
    return i


def U(x, z, N, l, M):
    sum = 0
    for i in range(1, N + 1):
        ro = np.pi * i / l
        A_n = np.sqrt(np.pi) * np.exp(-((np.pi * i) / 20) ** 2) * np.sin(np.pi * i / 2) * (
                special.erf((1j * np.pi * i + 100) / 20) - special.erf(
            (1j * np.pi * i - 100) / 20))
        sum += A_n * np.exp(M * ro ** 2 * z) * np.sin(ro * x)
    return sum / 2


def get_analytical(M, l, L, x_arr, z_arr, x_val, z_val, eps):
    n = n_min(eps)
    print(f'eps = {eps}')
    u_x = []
    for z in z_val:
        u_x.append([U(x, z, n, l, M) for x in x_arr])
    u_z = []
    for x in x_val:
        u_z.append([U(x, z, n, l, M) for z in z_arr])
    return u_x, u_z

# def get_analytical(M, l, L, x_array, z_array, eps):
#     # for i in eps_array:
#     #     print("z=", L / 3, " eps=", i, " Nmin=", F(i))
#     # for i in eps_array:
#     #     print("z=", 0.1 * 10 ** (-6), " eps=", i, " Nmin=", F(i), " Nexp=", Fexp(x_array, 10 * 10 ** (-6), F(i), i))
#     # y1 = [U(x, 0.00000001, n, l, M) for x in x_array]
#     # y2 = [U(x, L / 3, n, l, M) for x in x_array]
#     # y3 = [U(x, 2 * L / 3, n, l, M) for x in x_array]
#     # y4 = [U(x, L, n, l, M) for x in x_array]
#     # plt.plot(x_array, np.abs(y1), label="z = 0.01 мкм")
#     # plt.plot(x_array, np.abs(y2), label="z = 3.334 мкм")
#     # plt.plot(x_array, np.abs(y3), label="z = 6.667 мкм")
#     # plt.plot(x_array, np.abs(y4), label="z = 10 мкм")
#     #
#     # plt.xlabel("x")
#     # plt.ylabel("U(x, z)")
#     # plt.legend()
#     # plt.show()
#     #
#     # y1 = [U(0.00000001, z, n, l, M) for z in z_array]
#     # y2 = [U(3 * 10 ** (-6), z, n, l, M) for z in z_array]
#     # y3 = [U(6 * 10 ** (-6), z, n, l, M) for z in z_array]
#     # y4 = [U(10 * 10 ** (-6), z, n, l, M) for z in z_array]
#     #
#     # plt.plot(z_array, np.abs(y1), label="x = 0.01 мкм")
#     # plt.plot(z_array, np.abs(y2), label="x = 3 мкм")
#     # plt.plot(z_array, np.abs(y3), label="x = 6 мкм")
#     # plt.plot(z_array, np.abs(y4), label="x = 10 мкм")
#     #
#     # plt.xlabel("x")
#     # plt.ylabel("U(x, z)")
#     # plt.legend()
#     # plt.show()
#     n = n_min(eps)
#
#     return x_array, z_array
