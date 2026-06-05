import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv('DATA (1).csv')
X = df['x'].values.reshape(-1, 1)
y = df['y'].values

degrees = range(1, 20)
train_mse = []

for d in degrees:
    poly = PolynomialFeatures(degree=d, include_bias=True)
    X_poly = poly.fit_transform(X)

    model = LinearRegression(fit_intercept=False)
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)
    mse = mean_squared_error(y, y_pred)
    train_mse.append(mse)
    print(f"Degree {d:2d}: MSE = {mse:.4e}")

plt.figure(figsize=(10,5))
plt.plot(degrees, train_mse, 'bo-')
plt.xlabel('Степень полинома')
plt.ylabel('MSE')
plt.yscale('log')
plt.grid(True)
plt.title('Зависимость MSE от степени полинома')
plt.show()

best_degree = degrees[np.argmin(train_mse)]
best_mse = min(train_mse)
print(f"\nЛучшая степень: {best_degree}")
print(f"Минимальная MSE: {best_mse:.12f}")

poly_final = PolynomialFeatures(degree=best_degree, include_bias=True)
X_poly_final = poly_final.fit_transform(X)
model_final = LinearRegression(fit_intercept=False)
model_final.fit(X_poly_final, y)
y_pred_final = model_final.predict(X_poly_final)
final_mse = mean_squared_error(y, y_pred_final)

print(f"\nИТОГОВАЯ MSE (на всех 300 точках): {final_mse:.12f}")

plt.figure(figsize=(12,5))
plt.scatter(X, y, s=10, alpha=0.6, label='Истинные данные')
X_smooth = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
X_smooth_poly = poly_final.transform(X_smooth)
y_smooth = model_final.predict(X_smooth_poly)
plt.plot(X_smooth, y_smooth, 'r-', linewidth=2, label=f'Полином степени {best_degree}')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title(f'Линейная регрессия с полиномиальными признаками (MSE = {final_mse:.2e})')
plt.grid(True)
plt.show()

print(f"\nКоэффициенты w0...w{best_degree}:")
for i, coef in enumerate(model_final.coef_):
    print(f"  w{i} = {coef:.6f}")

# Получилось MSE = 8.165315853454