import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse.linalg import cg

X_columns = ['X11', 'X12', 'X21', 'X22', 'X31', 'X32']
Y_columns = ['Y1', 'Y2', 'Y3', 'Y4', 'Y5']

np.random.seed(42)
dataset = pd.DataFrame({
    "X11": np.round(np.random.uniform(0, 2, size=40), decimals=2),
    "X12": np.round(np.random.uniform(0, 1.5, size=40), decimals=2),
    "X21": np.round(np.random.uniform(0, 1.8, size=40), decimals=2),
    "X22": np.round(np.random.uniform(0, 1.9, size=40), decimals=2),
    "X31": np.round(np.random.uniform(0, 1.6, size=40), decimals=2),
    "X32": np.round(np.random.uniform(0, 2, size=40), decimals=2),
})

output_Y = pd.DataFrame({
    "Y1": np.round(2.1 * dataset["X11"] + 1.6 * dataset["X12"] +
                   np.random.uniform(-0.4, 0.4, size=40), decimals=2),
    "Y2": np.round(1.3 * dataset["X21"] + 1.9 * dataset["X22"] +
                   np.random.uniform(-0.3, 0.3, size=40), decimals=2),
    "Y3": np.round(1.4 * dataset["X31"] + 1.2 * dataset["X32"] +
                   np.random.uniform(-0.5, 0.5, size=40), decimals=2),
    "Y4": np.round(2.0 * dataset["X11"] + 1.5 * dataset["X21"] +
                   np.random.uniform(-0.5, 0.5, size=40), decimals=2),
    "Y5": np.round(1.7 * dataset["X12"] + 1.5 * dataset["X32"] +
                   np.random.uniform(-0.6, 0.6, size=40), decimals=2),
})

data = pd.concat([dataset, output_Y], axis=1)

data_X = data[X_columns].values
data_Y = data[Y_columns].values

Y_min = np.min(data_Y, axis=0)
Y_max = np.max(data_Y, axis=0)
norm_Y = (data_Y - Y_min) / (Y_max - Y_min)

A = np.dot(data_X.T, data_X)
coefficients = {}
aprox_Y = np.zeros_like(norm_Y)

for i, col in enumerate(Y_columns):
    b = np.dot(data_X.T, norm_Y[:, i])
    solution, info = cg(A, b)
    if info == 0:
        print(f"Розв'язок для {col} знайдено успішно.")
    elif info > 0:
        print(f"Метод не збігся після {info} ітерацій для {col}.")
    else:
        print(f"Метод спряжених напрямків не зміг знайти розв'язок для{col}.")
    coefficients[col] = solution
    aprox_Y[:, i] = np.dot(data_X, solution)


coefficients_df = pd.DataFrame(coefficients, index=X_columns)
print(coefficients_df)

plt.figure(figsize=(10, 5))
plt.plot(norm_Y[:, 0], label='Реальні значення', linestyle='-', color='green', linewidth=2)
plt.plot(aprox_Y[:, 0], label='Оцінені значення', linestyle='--', color='black', linewidth=2)
plt.xlabel('Зразок')
plt.ylabel('Нормалізоване значення')
plt.title('Порівняння реальних та оцінених значень')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.show()
