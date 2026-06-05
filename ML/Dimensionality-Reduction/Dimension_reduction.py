import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import requests
import time
import warnings
warnings.filterwarnings('ignore')

# Настройка визуализации
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)

# =============================================================================
# 1. ФУНКЦИЯ ЗАГРУЗКИ ДАННЫХ С ПОВТОРНЫМИ ПОПЫТКАМИ
# =============================================================================
def load_auto_data_with_retry(url, max_retries=3, timeout=10):
    """
    Загружает данные с обработкой ошибок и повторными попытками.
    Если основной источник недоступен — пробует альтернативы.
    """
    # Альтернативные источники (зеркала)
    alternative_urls = [
        "https://raw.githubusercontent.com/pbhatnagar/uci-datasets/master/auto-mpg/imports-85.data",
        "https://archive.ics.uci.edu/static/public/50/auto+imports.csv",  # новая версия архива
    ]

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

    # Список всех URL для перебора
    all_urls = [url] + alternative_urls

    for attempt in range(max_retries):
        for current_url in all_urls:
            try:
                print(f"🔄 Попытка #{attempt+1}: {current_url[:60]}...")
                response = requests.get(current_url, headers=headers, timeout=timeout)
                response.raise_for_status()

                # Парсим данные вручную для надёжности
                lines = response.text.strip().split('\n')
                data = [line.split(',') for line in lines if line.strip()]

                columns = [
                    'symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration',
                    'num-of-doors', 'body-style', 'drive-wheels', 'engine-location', 'wheel-base',
                    'length', 'width', 'height', 'curb-weight', 'engine-type', 'num-of-cylinders',
                    'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower',
                    'peak-rpm', 'city-mpg', 'highway-mpg', 'price'
                ]

                df = pd.DataFrame(data, columns=columns)
                # Преобразуем '?' в NaN и числовые колонки
                df = df.replace('?', np.nan)

                # Конвертируем числовые колонки
                numeric_candidates = ['symboling', 'normalized-losses', 'wheel-base', 'length',
                                     'width', 'height', 'curb-weight', 'engine-size', 'bore',
                                     'stroke', 'compression-ratio', 'horsepower', 'peak-rpm',
                                     'city-mpg', 'highway-mpg', 'price']

                for col in numeric_candidates:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                print(f"✅ Данные успешно загружены: {df.shape[0]} строк, {df.shape[1]} столбцов")
                return df

            except requests.exceptions.RequestException as e:
                print(f"⚠️ Ошибка загрузки: {type(e).__name__}: {e}")
                time.sleep(2 ** attempt)  # экспоненциальная задержка
                continue

    # Если все попытки провалились — создаём демо-данные для демонстрации кода
    print("❌ Не удалось загрузить данные. Создаю демо-датасет для демонстрации...")
    return create_demo_dataset()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

digits = load_digits()
X = digits.data
y = digits.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =============================================================================
# PCA АНАЛИЗ
# =============================================================================

pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_scaled)

print("="*100)
print("PCA РЕЗУЛЬТАТЫ")
print("="*100)
print(f"\nExplained variance ratio для первых 5 компонент:")
for i, var in enumerate(pca_full.explained_variance_ratio_[:5]):
    print(f"  PC{i+1}: {var:.4f} ({var*100:.2f}%)")
print(f"\nКумулятивная дисперсия для первых 5 компонент: {np.sum(pca_full.explained_variance_ratio_[:5]):.4f}")

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
scatter = plt.scatter(X_pca_full[:, 0], X_pca_full[:, 1], c=y, cmap='tab10', s=10, alpha=0.7)
plt.colorbar(scatter, label='Digit')
plt.xlabel(f'PC1 ({pca_full.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca_full.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('PCA: все признаки (64 компоненты)')

feature_variances = np.var(X, axis=0)
significant_features = np.where(feature_variances > np.percentile(feature_variances, 50))[0]
X_reduced = X[:, significant_features]
X_reduced_scaled = scaler.fit_transform(X_reduced)

print(f"\nКоличество признаков: исходное = {X.shape[1]}, после исключения = {X_reduced.shape[1]}")

pca_reduced = PCA()
X_pca_reduced = pca_reduced.fit_transform(X_reduced_scaled)

plt.subplot(1, 2, 2)
scatter = plt.scatter(X_pca_reduced[:, 0], X_pca_reduced[:, 1], c=y, cmap='tab10', s=10, alpha=0.7)
plt.colorbar(scatter, label='Digit')
plt.xlabel(f'PC1 ({pca_reduced.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca_reduced.explained_variance_ratio_[1]*100:.1f}%)')
plt.title(f'PCA: только значимые признаки ({X_reduced.shape[1]} features)')

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(np.cumsum(pca_full.explained_variance_ratio_), 'b-', label='Все признаки', linewidth=2)
plt.plot(np.cumsum(pca_reduced.explained_variance_ratio_), 'r-', label='Только значимые', linewidth=2)
plt.axhline(y=0.95, color='g', linestyle='--', label='95% variance')
plt.xlabel('Количество компонент')
plt.ylabel('Кумулятивная дисперсия')
plt.title('Сравнение кумулятивной дисперсии PCA')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# =============================================================================
# t-SNE АНАЛИЗ С ВАРЬИРОВАНИЕМ PERPLEXITY
# =============================================================================

print("\n" + "="*100)
print("t-SNE РЕЗУЛЬТАТЫ")
print("="*60)

perplexities = [10, 20, 30, 40, 50, 60, 70, 80, 90]
tsne_results = {}

print("Выполняется t-SNE для разных perplexity...")
print("(Это может занять некоторое время)")

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.ravel()

for idx, perp in enumerate(perplexities):
    print(f"  Обработка perplexity = {perp}...")

    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, n_iter=1000)
    X_tsne = tsne.fit_transform(X_scaled)
    tsne_results[perp] = X_tsne

    scatter = axes[idx].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=8, alpha=0.7)
    axes[idx].set_title(f'Perplexity = {perp}')
    axes[idx].set_xlabel('t-SNE 1')
    axes[idx].set_ylabel('t-SNE 2')
    axes[idx].grid(True, alpha=0.3)

plt.colorbar(scatter, ax=axes, label='Digit', ticks=range(10))
plt.suptitle('t-SNE визуализация датасета digits с разными значениями perplexity', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

print("\nСравнение лучших результатов (perplexity = 30, 40, 50):")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

best_perps = [30, 40, 50]
for idx, perp in enumerate(best_perps):
    X_tsne = tsne_results[perp]
    scatter = axes[idx].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=15, alpha=0.8)
    axes[idx].set_title(f't-SNE: perplexity = {perp}')
    axes[idx].set_xlabel('Component 1')
    axes[idx].set_ylabel('Component 2')
    axes[idx].grid(True, alpha=0.3)

plt.colorbar(scatter, ax=axes, label='Digit', ticks=range(10))
plt.suptitle('t-SNE результаты для оптимальных perplexity', fontsize=14)
plt.tight_layout()
plt.show()