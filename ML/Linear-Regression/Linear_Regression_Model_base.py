import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('data.csv')

print("Первые 5 строк данных:")
print(df.head())

print("\nИнформация о данных:")
print(df.info())

print("\nОсновные статистики:")
print(df.describe())

X = df['km'].values.reshape(-1, 1)  # пробег
y = df['price'].values              # цена

model = LinearRegression()
model.fit(X, y)

slope = model.coef_[0]      # коэффициент наклона (b1)
intercept = model.intercept_  # свободный член (b0)

print(f"\nУравнение регрессии: price = {slope:.4f} * km + {intercept:.2f}")

y_pred = model.predict(X)

mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print(f"\nМетрики качества модели:")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X, y, color='blue', alpha=0.6, label='Исходные данные')
plt.plot(X, y_pred, color='red', linewidth=2, label=f'Линия регрессии\ny = {slope:.4f}x + {intercept:.2f}')
plt.xlabel('Пробег (км)')
plt.ylabel('Цена (усл. ед.)')
plt.title('Линейная регрессия: цена vs пробег')
plt.legend()
plt.grid(True, alpha=0.3)

residuals = y - y_pred
plt.subplot(1, 2, 2)
plt.scatter(y_pred, residuals, color='green', alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
plt.xlabel('Предсказанные цены')
plt.ylabel('Остатки')
plt.title('График остатков')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

km_new = 100000
price_pred = model.predict([[km_new]])[0]
print(f"\nПрогноз цены для автомобиля с пробегом {km_new} км: {price_pred:.2f}")

results_df = pd.DataFrame({
    'Пробег (км)': X.flatten(),
    'Реальная цена': y,
    'Предсказанная цена': y_pred,
    'Ошибка': y - y_pred,
    'Относительная ошибка (%)': np.abs((y - y_pred) / y * 100)
})

print("\nПервые 10 наблюдений с предсказаниями:")
print(results_df.head(10).round(2))

correlation = np.corrcoef(X.flatten(), y)[0, 1]
print(f"\nКоэффициент корреляции Пирсона: {correlation:.4f}")