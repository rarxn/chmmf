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
