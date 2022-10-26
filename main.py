import numpy as np
import matplotlib.pyplot as plt

from analytical import get_analytical
from numerical import get_numerical

l = 10.  # 10mkm
L = 10.  # 10mkm

lbd = 2.  # 2mkm
n = 1
k = 2 * np.pi / lbd
M = 1j / (2 * k * n)

count = 500
eps = 10 ** -7


def create_plot(title, label):
    fig, ax = plt.subplots()
    ax.grid()
    ax.set_title(title)
    ax.set_xlabel(label)
    ax.set_ylabel('U(x,z)')
    return ax


def main():
    x_array = np.linspace(0, l, count)
    z_array = np.linspace(0, L, count)
    x_val = 2 * l / 3
    z_val = 2 * L / 3
    I_array = [20, 40, 80, 160, 320]
    K_array = [60, 120, 240, 480, 960]

    ax1, ax2 = create_plot(f'z={z_val:.4}', 'x, мкм'), create_plot(f'x={x_val:.4}', 'z, мкм')

    x, z = get_analytical(M, l, L, x_array, z_array, [x_val], [z_val], eps)
    ax1.plot(x_array, np.abs(x[0]), label='analytical')
    ax2.plot(z_array, np.abs(z[0]), label='analytical')

    for i, k in zip(I_array, K_array):
        x_array = np.linspace(0, l, i)
        z_array = np.linspace(0, L, k)
        x, z = get_numerical(M, l, L, x_array, z_array, [x_val], [z_val])
        ax1.plot(x_array, np.abs(x[0]), linestyle='--', label=f'I={i} K={k}')
        ax2.plot(z_array, np.abs(z[0]), linestyle='--', label=f'I={i} K={k}')
    ax1.legend()
    ax2.legend()
    plt.show()

    # x_values = [0.01 * 10 ** (-6),
    #             3. * 10 ** (-6),
    #             6. * 10 ** (-6),
    #             10. * 10 ** (-6)]
    # z_values = [0.01 * 10 ** (-6),
    #             3.334 * 10 ** (-6),
    #             6.667 * 10 ** (-6),
    #             10. * 10 ** (-6)]


if __name__ == '__main__':
    main()
