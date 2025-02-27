import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def f1(x):
    return 7 - np.exp(2 * x - 12)

def f2(x):
    return 3 + np.sqrt(x - 4)

x_min, x_max = 5, 8
step = 0.001
x_values = np.arange(x_min, x_max + step, step)

f1_star, f2_star = 6, 3

y_f1 = f1(x_values)
y_f2 = f2(x_values)

plt.figure(figsize=(8, 6))
plt.plot(x_values, y_f1, label='$f_1(x)$', color='blue')
plt.plot(x_values, y_f2, label='$f_2(x)$', color='red')
plt.axhline(y=f1_star, color='blue', linestyle='dashed', alpha=0.7, label='$f_1^*$')
plt.axhline(y=f2_star, color='red', linestyle='dashed', alpha=0.7, label='$f_2^*$')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.title('Графіки $f_1(x)$ та $f_2(x)$')
plt.legend()
plt.grid()
plt.show()

rows = []
for x in x_values:
    val_f1 = f1(x)
    val_f2 = f2(x)
    ratio_f1 = val_f1 / f1_star
    ratio_f2 = val_f2 / f2_star
    row_max = max(ratio_f1, ratio_f2)
    row_min = min(ratio_f1, ratio_f2)
    rows.append({
        'x': round(x, 4),
        'f1/f1*': round(ratio_f1, 4),
        'f2/f2*': round(ratio_f2, 4),
        'max(...)': round(row_max, 4),
        'min(...)': round(row_min, 4),
    })

df = pd.DataFrame(rows)

min_of_max = df['max(...)'].min()
df['--- (min of max)'] = df['max(...)'].apply(lambda v: str(v) if v == min_of_max else '-')

max_of_min = df['min(...)'].max()
df['--- (max of min)'] = df['min(...)'].apply(lambda v: str(v) if v == max_of_min else '-')

print(df.to_string(index=False))

rational_x = df.loc[(df['--- (min of max)'] != '-') & (df['--- (max of min)'] != '-'), 'x']
if not rational_x.empty:
    print(f'Як раціональний компроміс слід вибрати стратегію x = {rational_x.values[0]}')
else:
    print('Раціонального компромісу немає.')
