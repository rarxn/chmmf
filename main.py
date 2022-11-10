import numpy as np
import matplotlib.pyplot as plt

from tkinter import *
from tkinter import ttk
from analytical import get_analytical
from analytical import get_analytical1
from numerical import get_numerical

l = 10.  # 10mkm
L = 10.  # 10mkm

lbd = 2.  # 2mkm
n = 1.

k = 2 * np.pi / lbd
M = 1j / (2 * k * n)

x_val = 2 * l / 3
z_val = 2 * L / 3

I = 10
K = 5
numb = 5

count = 500
eps = 10 ** -7

table_data = []
choice = 1
var = 1

shag = [2, 4]


def calc_M():
    global k, M
    k = 2 * np.pi / lbd
    M = 1j / (2 * k * n)


def get_data(_l, _L, _lbd, _n, _x_val, _z_val, _choice, _I, _K, _numb, _var):
    global l, L, lbd, n, x_val, z_val, choice, I, K, numb, var, shag
    l = _l
    L = _L
    lbd = _lbd
    n = _n
    calc_M()
    x_val = _x_val
    z_val = _z_val
    I = _I
    K = _K
    numb = _numb
    choice = _choice
    var = _var
    if var == 1 or var == 3:
        shag = [2, 4]
    else:
        shag = [2, 2]


class Window:
    """Главное окно приложения"""

    def __init__(self, width, height, title):
        self.root = Tk()
        self.root.title(title)
        self.root.geometry(f"{width}x{height}+200+125")
        self.root.resizable(False, False)
        self.button = None
        self.button2 = None

        self.l = StringVar(value=str(l))
        self.L = StringVar(value=str(L))
        self.lbd = StringVar(value=str(lbd))
        self.n = StringVar(value=str(n))
        self.x_val = StringVar(value=f"{x_val:.5f}")
        self.z_val = StringVar(value=f"{z_val:.5f}")
        self.I = StringVar(value=str(I))
        self.K = StringVar(value=str(K))
        self.numb = StringVar(value=str(numb))
        self.choice = IntVar(value=str(choice))
        self.var = IntVar(value=str(var))

    def run(self):
        self.draw()
        self.root.mainloop()

    def draw(self):
        """Прорисовка объектов окна"""
        r = 0
        Label(self.root, text="l (мкм):").grid(row=r, column=0, pady=10, sticky=E)
        Entry(self.root, width=10, textvariable=self.l).grid(row=r, column=1)
        Label(self.root, text="L (мкм):").grid(row=r, column=2, padx=(10, 0), sticky=E)
        Entry(self.root, width=10, textvariable=self.L).grid(row=r, column=3)
        r += 1
        Label(self.root, text="lambda (мкм):").grid(row=r, column=0, pady=10, sticky=E)
        Entry(self.root, width=10, textvariable=self.lbd).grid(row=r, column=1)
        Label(self.root, text="n:").grid(row=r, column=2, padx=(10, 0), sticky=E)
        Entry(self.root, width=10, textvariable=self.n).grid(row=r, column=3)
        r += 1
        Label(self.root, text="x_val (мкм):").grid(row=r, column=0, pady=10, sticky=E)
        Entry(self.root, width=10, textvariable=self.x_val).grid(row=r, column=1)
        Label(self.root, text="z_val (мкм):").grid(row=r, column=2, padx=(10, 0), sticky=E)
        Entry(self.root, width=10, textvariable=self.z_val).grid(row=r, column=3)
        r += 1
        Label(self.root, text="I_начальное:").grid(row=r, column=0, pady=10, sticky=E)
        Entry(self.root, width=10, textvariable=self.I).grid(row=r, column=1)
        Label(self.root, text="K_начальное:").grid(row=r, column=2, padx=(10, 0), sticky=E)
        Entry(self.root, width=10, textvariable=self.K).grid(row=r, column=3)
        r += 1
        Label(self.root, text="число сеток:").grid(row=r, column=0, pady=10, sticky=E)
        Entry(self.root, width=10, textvariable=self.numb).grid(row=r, column=1)
        r += 1
        Radiobutton(self.root, text="Сходимость", variable=self.choice, value=1).grid(row=r, column=0, columnspan=2)
        Radiobutton(self.root, text="Численное решение", variable=self.choice, value=2).grid(row=r, column=2,
                                                                                             columnspan=2)
        r += 1
        Radiobutton(self.root, text="Никитин", variable=self.var, value=1).grid(row=r, column=0)
        Radiobutton(self.root, text="Яшакин", variable=self.var, value=2).grid(row=r, column=1, columnspan=2)
        Radiobutton(self.root, text="Каспаров", variable=self.var, value=3).grid(row=r, column=3)
        r += 1
        self.button = Button(self.root, text="Построить", command=self.start)
        self.button.grid(row=r, column=0, columnspan=2)
        self.button2 = Button(self.root, text="Таблица", command=self.open_table)
        self.button2.grid(row=r, column=2, columnspan=2)

    def start(self):
        self.button["state"] = DISABLED
        get_data(float(self.l.get()), float(self.L.get()), float(self.lbd.get()), float(self.n.get()),
                 float(self.x_val.get()), float(self.z_val.get()), int(self.choice.get()), int(self.I.get()),
                 int(self.K.get()), int(self.numb.get()), int(self.var.get()))
        print(
            f"start with l={l}, L={L}, lambda={lbd}, n={n}, x_val={x_val}, z_val={z_val}, I_start={I}, K_start={K},\n choice={choice}, numb={numb}, var={var}, shag={shag}")
        if choice == 1:
            convergence()
        else:
            numerical()
        self.button["state"] = NORMAL

    def open_table(self):
        get_data(float(self.l.get()), float(self.L.get()), float(self.lbd.get()), float(self.n.get()),
                 float(self.x_val.get()), float(self.z_val.get()), int(self.choice.get()), int(self.I.get()),
                 int(self.K.get()), int(self.numb.get()), int(self.var.get()))
        print(
            f"start with l={l}, L={L}, lambda={lbd}, n={n}, x_val={x_val}, z_val={z_val}, I_start={I}, K_start={K},\n choice={choice}, numb={numb}, var={var}, shag={shag}")
        accuracy()
        w = Toplevel(self.root)
        w.title('Таблица')
        w.resizable(False, False)
        w.attributes('-toolwindow', True)
        w.attributes('-topmost', True)
        w.grab_set()
        w.focus_set()
        table = ttk.Treeview(w, columns=5, show='headings')
        heads = ['I', 'K', 'E_hz_hx', f'E_hz/{shag[1]}_hx/{shag[0]}', 'delta_hz_hx']
        table['columns'] = heads
        for header in heads:
            table.heading(header, text=header, anchor='center')
            table.column(header, anchor='center')
        for row in table_data:
            table.insert('', END, values=row)
        table.pack()


def create_plot(title, label):
    fig, ax = plt.subplots()
    ax.grid()
    ax.set_title(title)
    ax.set_xlabel(label)
    ax.set_ylabel('U(x,z)')
    return ax


def accuracy():
    I_array = [I * (shag[0] ** i) for i in range(numb + 1)]
    K_array = [K * (shag[1] ** k) for k in range(numb + 1)]
    num_array = []
    an_array = []
    for i, k in zip(I_array, K_array):
        x_array = np.linspace(0, l, i + 1)
        z_array = np.linspace(0, L, k + 1)
        x, _ = get_numerical(M, l, L, x_array, z_array, [x_val], [z_val], var)
        num_array.append(x[0])
        an_array.append(get_analytical1(M, l, L, x_array, z_val, eps))

    global table_data
    table_data = []
    a = np.max(np.abs(np.abs(num_array[0]) - np.abs(an_array[0])))
    for i in range(1, len(I_array)):
        b = np.max(np.abs(np.abs(num_array[i]) - np.abs(an_array[i])))
        c = a / b
        print(I_array[i - 1], K_array[i - 1], a, b, c)
        table_data.append([I_array[i - 1], K_array[i - 1], a, b, c])
        a = b


def convergence():
    x_array = np.linspace(0, l, count)
    z_array = np.linspace(0, L, count)
    I_array = [I * (shag[0] ** i) for i in range(numb)]
    K_array = [K * (shag[1] ** k) for k in range(numb)]

    ax1, ax2 = create_plot(f'z={z_val:.4}', 'x, мкм'), create_plot(f'x={x_val:.4}', 'z, мкм')

    x, z = get_analytical(M, l, L, x_array, z_array, [x_val], [z_val], eps)
    ax1.plot(x_array, np.abs(x[0]), label='analytical')
    ax2.plot(z_array, np.abs(z[0]), label='analytical')
    for i, k in zip(I_array, K_array):
        x_array = np.linspace(0, l, i + 1)
        z_array = np.linspace(0, L, k + 1)
        x, z = get_numerical(M, l, L, x_array, z_array, [x_val], [z_val], var)
        ax1.plot(x_array, np.abs(x[0]), linestyle='--', label=f'I={i} K={k}')
        ax2.plot(z_array, np.abs(z[0]), linestyle='--', label=f'I={i} K={k}')
    ax1.legend()
    ax2.legend()
    plt.show()


def numerical():
    x_array = np.linspace(0, l, 200)
    z_array = np.linspace(0, L, 400)
    x_values = [0.01, 3., 6., l]
    z_values = [0.01, L / 3, 2 * L / 3, L]
    ax1, ax2 = create_plot('', 'x, мкм'), create_plot('', 'z, мкм')
    x, z = get_numerical(M, l, L, x_array, z_array, x_values, z_values, var)

    [ax1.plot(x_array, np.abs(_x), label=f'z= {val:.4} мкм') for _x, val in zip(x, z_values)]
    ax1.legend(loc='upper right')

    [ax2.plot(z_array, np.abs(_z), label=f'x= {val:.4} мкм') for _z, val in zip(z, x_values)]
    ax2.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    # accuracy()
    window = Window(320, 290, 'model')
    window.run()
