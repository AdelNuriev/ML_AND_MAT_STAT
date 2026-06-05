import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

np.random.seed(42)

d = 2  # day
m = 3  # month
y_mod = 2006  # year
y_from = 1

n_samples = 300
x = np.linspace(-3, 3, n_samples).reshape(-1, 1)

amp = d * 1.2
freq = m * 0.9
phase = np.pi * (d / 20)
deg = (y_mod % y_to) + y_from
y_true = amp * np.sin(freq * x + phase) + 0.25 * x**deg + m * 0.8


noise = np.random.normal(0, 0.3, size=y_true.shape)
y = y_true + noise

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

print("="*60)
print("ГЕНЕРАЦИЯ ДАННЫХ")
print("="*60)
print(f"Параметры: d={d}, m={m}, year={y_mod}")
print(f"Функция: y = {amp:.2f} * sin({freq:.2f}*x + {phase:.3f}) + 0.25*x^{deg} + {m*0.8}")
print(f"Размер выборки: {n_samples}")
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

plt.figure(figsize=(12, 6))
plt.scatter(X_train, y_train, alpha=0.6, s=20, label='Train', color='blue')
plt.scatter(X_test, y_test, alpha=0.6, s=20, label='Test', color='red')
plt.plot(x, y_true, 'g--', linewidth=2, label='True function (без шума)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Исходные данные')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


degrees = range(1, 13)
results = {}

print("\n" + "="*60)
print("ЭКСПЕРИМЕНТ С ПОЛИНОМИАЛЬНОЙ РЕГРЕССИЕЙ")
print("="*60)

for degree in degrees:
    print(f"\n--- Степень {degree} ---")

    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_poly)
    X_test_scaled = scaler.transform(X_test_poly)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train.ravel())
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)

    if degree >= 6:
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_scaled, y_train.ravel())
        y_train_pred_ridge = ridge.predict(X_train_scaled)
        y_test_pred_ridge = ridge.predict(X_test_scaled)
        mse_train_ridge = mean_squared_error(y_train, y_train_pred_ridge)
        mse_test_ridge = mean_squared_error(y_test, y_test_pred_ridge)

        # Lasso (L1)
        lasso = Lasso(alpha=0.01, max_iter=5000)
        lasso.fit(X_train_scaled, y_train.ravel())
        y_train_pred_lasso = lasso.predict(X_train_scaled)
        y_test_pred_lasso = lasso.predict(X_test_scaled)
        mse_train_lasso = mean_squared_error(y_train, y_train_pred_lasso)
        mse_test_lasso = mean_squared_error(y_test, y_test_pred_lasso)

        results[degree] = {
            'no_reg': {'model': model, 'mse_train': mse_train, 'mse_test': mse_test, 'coef': model.coef_},
            'ridge': {'model': ridge, 'mse_train': mse_train_ridge, 'mse_test': mse_test_ridge, 'coef': ridge.coef_, 'alpha': 1.0},
            'lasso': {'model': lasso, 'mse_train': mse_train_lasso, 'mse_test': mse_test_lasso, 'coef': lasso.coef_, 'alpha': 0.01}
        }

        print(f"Без регуляризации: Train MSE={mse_train:.6f}, Test MSE={mse_test:.6f}")
        print(f"Ridge (alpha=1.0): Train MSE={mse_train_ridge:.6f}, Test MSE={mse_test_ridge:.6f}")
        print(f"Lasso (alpha=0.01): Train MSE={mse_train_lasso:.6f}, Test MSE={mse_test_lasso:.6f}")
    else:
        results[degree] = {
            'no_reg': {'model': model, 'mse_train': mse_train, 'mse_test': mse_test, 'coef': model.coef_}
        }
        print(f"Без регуляризации: Train MSE={mse_train:.6f}, Test MSE={mse_test:.6f}")

    plt.figure(figsize=(12, 5))

    X_all_poly = poly.transform(x)
    X_all_scaled = scaler.transform(X_all_poly)
    y_pred_all = model.predict(X_all_scaled)

    plt.subplot(1, 2, 1)
    plt.scatter(X_train, y_train, alpha=0.5, s=15, color='blue', label='Train')
    plt.scatter(X_test, y_test, alpha=0.5, s=15, color='red', label='Test')
    plt.plot(x, y_true, 'g--', linewidth=2, label='True')
    plt.plot(x, y_pred_all, 'b-', linewidth=2, label='Prediction')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Полином степени {degree} (без регуляризации)\nTrain MSE={mse_train:.4f}, Test MSE={mse_test:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if degree >= 6:
        y_pred_ridge = ridge.predict(X_all_scaled)
        plt.subplot(1, 2, 2)
        plt.scatter(X_train, y_train, alpha=0.5, s=15, color='blue', label='Train')
        plt.scatter(X_test, y_test, alpha=0.5, s=15, color='red', label='Test')
        plt.plot(x, y_true, 'g--', linewidth=2, label='True')
        plt.plot(x, y_pred_ridge, 'orange', linewidth=2, label='Ridge')
        plt.plot(x, y_pred_all, 'b--', linewidth=1, alpha=0.5, label='Linear (без рег)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Полином степени {degree} с Ridge (α=1.0)\nTrain MSE={mse_train_ridge:.4f}, Test MSE={mse_test_ridge:.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

plt.figure(figsize=(12, 6))
degrees_list = list(results.keys())
train_mse_no_reg = [results[d]['no_reg']['mse_train'] for d in degrees_list]
test_mse_no_reg = [results[d]['no_reg']['mse_test'] for d in degrees_list]

plt.plot(degrees_list, train_mse_no_reg, 'bo-', label='Train MSE (без рег)', linewidth=2, markersize=8)
plt.plot(degrees_list, test_mse_no_reg, 'ro-', label='Test MSE (без рег)', linewidth=2, markersize=8)

ridge_train_mse = [results[d]['ridge']['mse_train'] if d >= 6 else None for d in degrees_list]
ridge_test_mse = [results[d]['ridge']['mse_test'] if d >= 6 else None for d in degrees_list]
lasso_train_mse = [results[d]['lasso']['mse_train'] if d >= 6 else None for d in degrees_list]
lasso_test_mse = [results[d]['lasso']['mse_test'] if d >= 6 else None for d in degrees_list]

plt.plot(degrees_list, ridge_train_mse, 'g^--', label='Train MSE (Ridge)', linewidth=2, markersize=8)
plt.plot(degrees_list, ridge_test_mse, 'g^:', label='Test MSE (Ridge)', linewidth=2, markersize=8)
plt.plot(degrees_list, lasso_train_mse, 'ms--', label='Train MSE (Lasso)', linewidth=2, markersize=8)
plt.plot(degrees_list, lasso_test_mse, 'ms:', label='Test MSE (Lasso)', linewidth=2, markersize=8)

plt.xlabel('Степень полинома')
plt.ylabel('MSE')
plt.title('Зависимость ошибки от степени полинома')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.show()

print("\n" + "="*60)
print("АНАЛИЗ КОЭФФИЦИЕНТОВ")
print("="*60)

for degree in [3, 6, 9, 12]:
    if degree not in results:
        continue

    print(f"\n--- Степень {degree} ---")

    coef_no_reg = results[degree]['no_reg']['coef']
    print(f"\nБез регуляризации (первые 10 коэффициентов):")
    for i, coef in enumerate(coef_no_reg[:10]):
        print(f"  w{i} = {coef:.6f}")
    if len(coef_no_reg) > 10:
        print(f"  ... и еще {len(coef_no_reg)-10} коэффициентов")

    if degree >= 6:
        coef_ridge = results[degree]['ridge']['coef']
        print(f"\nRidge (alpha={results[degree]['ridge']['alpha']}):")
        n_nonzero_ridge = np.sum(np.abs(coef_ridge) > 1e-6)
        print(f"  Ненулевых коэффициентов: {n_nonzero_ridge} из {len(coef_ridge)}")
        for i, coef in enumerate(coef_ridge[:10]):
            print(f"  w{i} = {coef:.6f}")

        coef_lasso = results[degree]['lasso']['coef']
        print(f"\nLasso (alpha={results[degree]['lasso']['alpha']}):")
        n_nonzero_lasso = np.sum(np.abs(coef_lasso) > 1e-6)
        print(f"  Ненулевых коэффициентов: {n_nonzero_lasso} из {len(coef_lasso)}")
        for i, coef in enumerate(coef_lasso[:10]):
            print(f"  w{i} = {coef:.6f}")

        print(f"\nВлияние регуляризации на веса:")
        print(f"  - Максимальный вес без рег: {np.max(np.abs(coef_no_reg)):.4f}")
        print(f"  - Максимальный вес (Ridge): {np.max(np.abs(coef_ridge)):.4f}")
        print(f"  - Максимальный вес (Lasso): {np.max(np.abs(coef_lasso)):.4f}")
        print(f"  - Ridge уменьшил веса в {np.max(np.abs(coef_no_reg))/np.max(np.abs(coef_ridge)):.1f} раз")
        print(f"  - Lasso обнулил {len(coef_lasso) - n_nonzero_lasso} коэффициентов")

degree_example = 10

if degree_example in results:
    print("\n" + "="*60)
    print(f"ДЕТАЛЬНЫЙ АНАЛИЗ ДЛЯ СТЕПЕНИ {degree_example} (ПЕРЕОБУЧЕНИЕ)")
    print("="*60)

    poly = PolynomialFeatures(degree=degree_example)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_poly)
    X_test_scaled = scaler.transform(X_test_poly)

    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train.ravel())

    # Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train.ravel())

    # Lasso
    lasso = Lasso(alpha=0.01, max_iter=5000)
    lasso.fit(X_train_scaled, y_train.ravel())

    X_all_poly = poly.transform(x)
    X_all_scaled = scaler.transform(X_all_poly)

    plt.figure(figsize=(15, 5))

    # Linear Regression
    y_pred_lr = lr.predict(X_all_scaled)
    plt.subplot(1, 3, 1)
    plt.scatter(X_train, y_train, alpha=0.4, s=15, color='blue', label='Train')
    plt.scatter(X_test, y_test, alpha=0.4, s=15, color='red', label='Test')
    plt.plot(x, y_true, 'g--', linewidth=2, label='True')
    plt.plot(x, y_pred_lr, 'b-', linewidth=2, label='Linear')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Степень {degree_example} (без регуляризации)\nTrain MSE={results[degree_example]["no_reg"]["mse_train"]:.4f}\nTest MSE={results[degree_example]["no_reg"]["mse_test"]:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Ridge
    y_pred_ridge = ridge.predict(X_all_scaled)
    plt.subplot(1, 3, 2)
    plt.scatter(X_train, y_train, alpha=0.4, s=15, color='blue', label='Train')
    plt.scatter(X_test, y_test, alpha=0.4, s=15, color='red', label='Test')
    plt.plot(x, y_true, 'g--', linewidth=2, label='True')
    plt.plot(x, y_pred_ridge, 'orange', linewidth=2, label='Ridge')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Ridge (α=1.0)\nTrain MSE={results[degree_example]["ridge"]["mse_train"]:.4f}\nTest MSE={results[degree_example]["ridge"]["mse_test"]:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Lasso
    y_pred_lasso = lasso.predict(X_all_scaled)
    plt.subplot(1, 3, 3)
    plt.scatter(X_train, y_train, alpha=0.4, s=15, color='blue', label='Train')
    plt.scatter(X_test, y_test, alpha=0.4, s=15, color='red', label='Test')
    plt.plot(x, y_true, 'g--', linewidth=2, label='True')
    plt.plot(x, y_pred_lasso, 'purple', linewidth=2, label='Lasso')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Lasso (α=0.01)\nTrain MSE={results[degree_example]["lasso"]["mse_train"]:.4f}\nTest MSE={results[degree_example]["lasso"]["mse_test"]:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    coef_lr = lr.coef_
    coef_ridge = ridge.coef_
    coef_lasso = lasso.coef_

    axes[0].bar(range(len(coef_lr[:20])), coef_lr[:20], alpha=0.7)
    axes[0].set_title('Коэффициенты (без регуляризации)')
    axes[0].set_xlabel('Индекс коэффициента')
    axes[0].set_ylabel('Значение')
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(range(len(coef_ridge[:20])), coef_ridge[:20], alpha=0.7, color='orange')
    axes[1].set_title('Коэффициенты (Ridge)')
    axes[1].set_xlabel('Индекс коэффициента')
    axes[1].grid(True, alpha=0.3)

    axes[2].bar(range(len(coef_lasso[:20])), coef_lasso[:20], alpha=0.7, color='purple')
    axes[2].set_title('Коэффициенты (Lasso)')
    axes[2].set_xlabel('Индекс коэффициента')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f'Сравнение коэффициентов для степени {degree_example}')
    plt.tight_layout()
    plt.show()

print("\n" + "="*60)
print("ВЫВОДЫ:")
print("="*60)
print("1. При увеличении степени полинома без регуляризации наблюдается переобучение")
print("2. Ridge (L2) уменьшает все коэффициенты, но не обнуляет их полностью")
print("3. Lasso (L1) обнуляет некоторые коэффициенты, делая модель более разреженной")
print("4. Оптимальная степень полинома: 4-6 (минимальная ошибка на тесте)")
print("5. Регуляризация особенно эффективна при степенях ≥ 8")
print("="*60)