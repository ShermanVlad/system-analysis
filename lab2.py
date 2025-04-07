import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def f1(x1, x2):
    return 4*x1**3 - x1**2*x2**2 + 2*x2**2 - 10

def f2(x1, x2):
    return 4*x1**2 - 6*x1*x2 + 2*x2**2 + 5

a1 = 0
b1 = 3
a2 = 0
b2 = 4
h = 0.01

x11 = []
x21 = []
f12 = []
Tablef12 = []

d1 = a1
while d1 <= b1:
    curr_mean = []
    d2 = a2
    while d2 <= b2:
        x11.append(d1)
        x21.append(d2)
        curr_mean.append(f1(d1, d2))
        Tablef12.append(f1(d1, d2))
        d2 += h
    f12.append(min(curr_mean))
    d1 += h

x12 = []
x22 = []
f21 = []
Tablef21 = []

d1 = a1
while d1 <= b1:
    curr_mean = []
    d2 = a2
    while d2 <= b2:
        x12.append(d2)
        x22.append(d1)
        curr_mean.append(f2(d2, d1))
        Tablef21.append(f2(d2, d1))
        d2 += h
    f21.append(min(curr_mean))
    d1 += h

try:
    data = pd.DataFrame({
        'x1': x11[:len(Tablef12)],
        'x2': x21[:len(Tablef12)],
        'f1': Tablef12,
        'f2': Tablef21[:len(Tablef12)]
    })
    data.to_excel('results_21.xlsx', index=False)
except ModuleNotFoundError:
    data.to_csv('results_21.csv', index=False)

f1z = round(max(f12), 3)
f2z = round(max(f21), 3)
print(f"Максимум f1: {f1z}, Максимум f2: {f2z}")

x = np.linspace(a2, b2, 1000)
plt.figure(figsize=(10, 6))
for x1_val in np.linspace(a1, b1, 5):
    y = [f1(x1_val, x[i]) for i in range(len(x))]
    plt.plot(x, y, label=f'x1={round(x1_val,1)}')
plt.title('Функція f1(x1,x2) при різних значеннях x1')
plt.xlabel('x2')
plt.ylabel('f1(x1,x2)')
plt.legend()
plt.grid()
plt.show()

x = np.linspace(a1, b1, 1000)
plt.figure(figsize=(10, 6))
for x2_val in np.linspace(a2, b2, 5):
    y = [f2(x[i], x2_val) for i in range(len(x))]
    plt.plot(x, y, label=f'x2={round(x2_val,1)}')
plt.title('Функція f2(x1,x2) при різних значеннях x2')
plt.xlabel('x1')
plt.ylabel('f2(x1,x2)')
plt.legend()
plt.grid()
plt.show()

d1 = a1
x_opt = []
y_opt = []
h = 0.01

while d1 <= b1 + h:
    d2 = a2
    while d2 <= b2 + h:
        if (f1(round(d1, 3), round(d2, 3)) >= f1z and
                f2(round(d1, 3), round(d2, 3)) >= f2z):
            x_opt.append(round(d1, 3))
            y_opt.append(round(d2, 3))
        d2 += h
    d1 += h

plt.figure(figsize=(8, 6))
plt.scatter(x_opt, y_opt)
plt.title('Оптимальні точки')
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid()
plt.show()

for i in range(len(x_opt)):
    delta = max(f1(x_opt[i], y_opt[i]) - f1z, f2(x_opt[i], y_opt[i]) - f2z)
    if delta < h:
        print(f"Оптимальна точка: ({x_opt[i]}, {y_opt[i]})")