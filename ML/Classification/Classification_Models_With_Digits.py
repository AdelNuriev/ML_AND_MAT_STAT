import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

digits = load_digits()
X = digits.data
y = digits.target

plt.figure(figsize=(20, 4))
for index, (image, label) in enumerate(zip(X[0:5], y[0:5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (8, 8)), cmap=plt.cm.gray)
    plt.title('Ответ: %i\n' % label, fontsize=14)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

print(f"Размер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")
print(f"Количество классов: {len(np.unique(y))}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

classifiers = {
    'SVM': {
        'model': SVC(random_state=42),
        'params': {
            'C': [0.1, 1, 10, 100],
            'gamma': [0.001, 0.01, 0.1, 1],
            'kernel': ['rbf']
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        }
    },
    'KNN': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance']
        }
    }
}

best_model = None
best_score = 0
best_name = ""
results = {}

print("\n" + "=" * 60)
print("ПОИСК ЛУЧШЕГО КЛАССИФИКАТОРА")
print("=" * 60)

for name, clf_dict in classifiers.items():
    print(f"\n--- {name} ---")
    grid = GridSearchCV(
        clf_dict['model'],
        clf_dict['params'],
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    grid.fit(X_train_scaled, y_train)

    y_pred = grid.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results[name] = {
        'best_params': grid.best_params_,
        'test_accuracy': acc,
        'model': grid.best_estimator_,
        'predictions': y_pred
    }
    print(f"Лучшие параметры: {grid.best_params_}")
    print(f"Точность на тесте: {acc:.4f}")

    if acc > best_score:
        best_score = acc
        best_model = grid.best_estimator_
        best_name = name
        best_predictions = y_pred

print("\n" + "=" * 60)
print(f"ЛУЧШИЙ КЛАССИФИКАТОР: {best_name}")
print(f"Точность: {best_score:.4f}")
print(f"Параметры: {results[best_name]['best_params']}")
print("=" * 60)

errors = (best_predictions != y_test)
error_indices = np.where(errors)[0]

print(f"\nКоличество ошибок: {len(error_indices)} из {len(y_test)} ({100 * len(error_indices) / len(y_test):.1f}%)")

cm = confusion_matrix(y_test, best_predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=digits.target_names,
            yticklabels=digits.target_names)
plt.title(f'Матрица ошибок - {best_name} (точность: {best_score:.4f})')
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
plt.show()

if len(error_indices) > 0:
    n_errors = min(len(error_indices), 20)
    n_cols = 5
    n_rows = (n_errors + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    axes = axes.flatten()

    for i, idx in enumerate(error_indices[:n_errors]):
        axes[i].imshow(X_test[idx].reshape(8, 8), cmap='gray')
        axes[i].set_title(f"True: {y_test[idx]} | Pred: {best_predictions[idx]}", fontsize=10)
        axes[i].axis('off')

    for j in range(n_errors, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f'Ошибки классификатора {best_name} (всего: {len(error_indices)})', fontsize=14)
    plt.tight_layout()
    plt.show()
else:
    print("Идеальная классификация! Ошибок нет.")

print("\n" + "=" * 60)
print("ПРИМЕНЕНИЕ t-SNE ДЛЯ ПРЕДОБРАБОТКИ")
print("=" * 60)

X_scaled = scaler.fit_transform(X)

n_components_list = [2, 10, 20, 30, 40]
tsne_accuracies = {}

for n_comp in n_components_list:
    print(f"\n--- t-SNE с {n_comp} компонентами ---")

    tsne = TSNE(n_components=n_comp, random_state=42, perplexity=30, n_iter=1000)
    X_tsne = tsne.fit_transform(X_scaled)

    X_train_tsne = X_tsne[X_train.index] if hasattr(X_train, 'index') else X_tsne[:len(X_train)]
    X_test_tsne = X_tsne[len(X_train):]

    if best_name == 'SVM':
        model_tsne = SVC(**results[best_name]['best_params'], random_state=42)
    elif best_name == 'Random Forest':
        model_tsne = RandomForestClassifier(**results[best_name]['best_params'], random_state=42)
    else:
        model_tsne = KNeighborsClassifier(**results[best_name]['best_params'])

    model_tsne.fit(X_train_tsne, y_train)
    y_pred_tsne = model_tsne.predict(X_test_tsne)
    acc_tsne = accuracy_score(y_test, y_pred_tsne)
    tsne_accuracies[n_comp] = acc_tsne
    print(f"Точность: {acc_tsne:.4f}")

tsne_2d = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne_2d = tsne_2d.fit_transform(X_scaled)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_tsne_2d[:, 0], X_tsne_2d[:, 1], c=y, cmap='tab10', s=20, alpha=0.7)
plt.colorbar(scatter, label='Цифра', ticks=range(10))
plt.title('t-SNE проекция датасета digits (2D)')
plt.xlabel('t-SNE компонента 1')
plt.ylabel('t-SNE компонента 2')
plt.show()

print("\n" + "=" * 60)
print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
print("=" * 60)
print(f"{'Классификатор':<20} {'Без t-SNE':<15} {'Лучший t-SNE':<15} {'Изменение':<10}")
print("-" * 60)

for name in results:
    acc_original = results[name]['test_accuracy']
    best_tsne_for_model = 0
    best_n_for_model = None
    for n_comp, acc_list in tsne_accuracies.items():
        if name == 'SVM':
            temp_model = SVC(**results[name]['best_params'], random_state=42)
        elif name == 'Random Forest':
            temp_model = RandomForestClassifier(**results[name]['best_params'], random_state=42)
        else:
            temp_model = KNeighborsClassifier(**results[name]['best_params'])

        tsne_temp = TSNE(n_components=n_comp, random_state=42, perplexity=30, n_iter=1000)
        X_tsne_temp = tsne_temp.fit_transform(X_scaled)
        X_train_temp = X_tsne_temp[:len(X_train)]
        X_test_temp = X_tsne_temp[len(X_train):]
        temp_model.fit(X_train_temp, y_train)
        acc_temp = accuracy_score(y_test, temp_model.predict(X_test_temp))

        if acc_temp > best_tsne_for_model:
            best_tsne_for_model = acc_temp
            best_n_for_model = n_comp

    change = best_tsne_for_model - acc_original
    print(f"{name:<20} {acc_original:<15.4f} {best_tsne_for_model:<15.4f} {change:+10.4f}")

print("\n" + "=" * 60)
print(f"ИТОГ: Лучший классификатор - {best_name} с точностью {best_score:.4f}")
print(f"После t-SNE максимальная точность: {max(tsne_accuracies.values()):.4f}")
print("=" * 60)

print("\n" + classification_report(y_test, best_predictions, target_names=digits.target_names.astype(str)))