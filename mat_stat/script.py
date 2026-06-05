from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

df = pd.read_csv('job_salary_prediction_dataset.csv')

print("=" * 80)
print("Анализ данных о зарплатах IT-специалистов")
print("=" * 80)

print("\n" + "=" * 80)
print("1. ОСНОВНАЯ ИНФОРМАЦИЯ О ДАННЫХ")
print("=" * 80)

n_observations = df.shape[0]
n_variables = df.shape[1]
print(f"\nКоличество наблюдений (строк): {n_observations}")
print(f"Количество переменных (столбцов): {n_variables}")

print("\nТипы данных переменных:")
print(df.dtypes)

print("\nКоличество пропущенных значений по столбцам:")
print(df.isnull().sum())

print("\n" + "=" * 80)
print("2. УДАЛЕНИЕ ПРОПУЩЕННЫХ ЗНАЧЕНИЙ")
print("=" * 80)

df_clean = df.dropna()
print(f"Размер данных до удаления пропусков: {df.shape}")
print(f"Размер данных после удаления пропусков: {df_clean.shape}")

print("\n" + "=" * 80)
print("3. СТАТИСТИЧЕСКИЕ ХАРАКТЕРИСТИКИ ЧИСЛОВЫХ СТОЛБЦОВ")
print("=" * 80)

numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
print(f"\nЧисловые столбцы: {numeric_cols}")

for col in numeric_cols:
    print(f"\n--- {col} ---")
    data = df_clean[col]

    mean_val = data.mean()
    var_val = data.var()
    min_val = data.min()
    max_val = data.max()
    q1 = data.quantile(0.25)
    q2 = data.quantile(0.50)
    q3 = data.quantile(0.75)

    print(f"  Среднее значение: {mean_val:.2f}")
    print(f"  Дисперсия: {var_val:.2f}")
    print(f"  Минимум: {min_val:.2f}")
    print(f"  Максимум: {max_val:.2f}")
    print(f"  25-й процентиль (Q1): {q1:.2f}")
    print(f"  50-й процентиль (Медиана): {q2:.2f}")
    print(f"  75-й процентиль (Q3): {q3:.2f}")

print("\n" + "-" * 40)
print("Корреляционная матрица (числовые столбцы):")
correlation_matrix = df_clean[numeric_cols].corr()
print(correlation_matrix.round(4))

if 'salary' in numeric_cols:
    print("\nКорреляции с зарплатой (salary):")
    salary_corr = correlation_matrix['salary'].sort_values(ascending=False)
    for col, corr in salary_corr.items():
        if col != 'salary':
            print(f"  {col}: {corr:.4f}")

print("\n" + "=" * 80)
print("5. ПОИСК ВЫБРОСОВ МЕТОДОМ МЕЖКВАРТИЛЬНОГО РАЗМАХА (IQR)")
print("=" * 80)

for col in numeric_cols:
    print(f"\n--- Анализ выбросов для признака: {col} ---")

    data = df_clean[col]
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    print(f"  Q1 (25-й процентиль): {Q1:.2f}")
    print(f"  Q3 (75-й процентиль): {Q3:.2f}")
    print(f"  Межквартильный размах (IQR): {IQR:.2f}")
    print(f"  Нижняя граница: {lower_bound:.2f}")
    print(f"  Верхняя граница: {upper_bound:.2f}")

    outliers_lower = data[data < lower_bound]
    outliers_upper = data[data > upper_bound]
    total_outliers = len(outliers_lower) + len(outliers_upper)
    outliers_percent = (total_outliers / len(data)) * 100

    print(f"  Количество выбросов снизу: {len(outliers_lower)}")
    print(f"  Количество выбросов сверху: {len(outliers_upper)}")
    print(f"  Всего выбросов: {total_outliers} ({outliers_percent:.2f}% от данных)")

    if total_outliers > 0 and col == 'salary':
        print(f"\n  Примеры выбросов по зарплате:")
        outlier_salaries = df_clean[df_clean[col] > upper_bound][
            ['job_title', 'experience_years', 'location', 'salary']].head(10)
        print(outlier_salaries.to_string(index=False))

print("\nВизуализация выбросов на боксплоте...")
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_clean, y='salary', color='lightcoral')
plt.title('Боксплот зарплаты с визуализацией выбросов', fontweight='bold')
plt.ylabel('Зарплата')
plt.tight_layout()
plt.savefig('outliers_boxplot.png', dpi=150, bbox_inches='tight')
plt.show()
print("График выбросов сохранен в файл 'outliers_boxplot.png'")

alpha = 0.05
education_groups = [df[df['education_level'] == edu]['salary'].values
                    for edu in df['education_level'].unique()]

f_stat, p_value = f_oneway(*education_groups)

print(f"  Нулевая гипотеза H0: средние зарплаты одинаковы для всех уровней образования")
print(f"  Альтернативная гипотеза H1: хотя бы одна группа отличается")
print(f"  Результаты теста:")
print(f"    F-статистика: {f_stat:.4f}")
print(f"    p-value: {0.023657}")

if p_value < alpha:
    print(f"  ВЫВОД: p-value ({0.023657}) < {alpha}. Отвергаем H0.")

    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    tukey_data = df[['salary', 'education_level']].copy()
    tukey_result = pairwise_tukeyhsd(tukey_data['salary'], tukey_data['education_level'], alpha=0.05)
    print("\n  Пост-хок анализ (Tukey HSD):")
    print(tukey_result)
else:
    print(f"  ВЫВОД: p-value ({p_value:.6f}) >= {alpha}. Нет оснований отвергать H0.")
    print(f"  → Статистически значимых различий в зарплатах нет.")

X = df[['experience_years']]
y = df['salary']

model = LinearRegression()
model.fit(X, y)

df['предсказание'] = model.predict(X)

print(f"Коэффициент наклона (slope): {model.coef_[0]}")
print(f"Сдвиг (intercept): {model.intercept_}")
print(df)

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

plt.scatter(df['experience_years'], df['salary'],
            color='royalblue', alpha=0.7, edgecolors='k', label='Реальные данные')

df_sorted = df.sort_values(by='experience_years')
plt.plot(df_sorted['experience_years'], df_sorted['предсказание'],
         color='crimson', linewidth=3, label=f'Линия регрессии (y = {model.coef_[0]:.2f}*x + {model.intercept_:.2f})')

plt.title('Зависимость зарплаты от количества сертификатов', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Количество сертификатов (experience_years)', fontsize=12)
plt.ylabel('Зарплата (salary)', fontsize=12)

plt.legend(fontsize=11, loc='upper left')
plt.tight_layout()
plt.savefig('regression_plot.png', dpi=300, bbox_inches='tight')