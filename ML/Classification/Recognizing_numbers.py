import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.ndimage import label
from matplotlib.image import imread

digits = load_digits()
X, y = digits.data, digits.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=2, random_state=42)
model.fit(X_scaled, y)

print("Модель обучена на датасете digits")
print(f"Точность на обучающей выборке: {model.score(X_scaled, y):.4f}")


def preprocess_single_digit(cropped_image, target_size=(8, 8)):
    from skimage.transform import resize

    if len(cropped_image.shape) == 3:
        cropped_image = np.dot(cropped_image[..., :3], [0.299, 0.587, 0.114])

    resized = resize(cropped_image, target_size, mode='constant', anti_aliasing=True)
    digit_image = resized * 16
    return digit_image.flatten()


def extract_digits_from_image(image, min_area=20):
    if len(image.shape) == 3:
        gray = np.dot(image[..., :3], [0.299, 0.587, 0.114])
    else:
        gray = image.copy()

    binary = (gray > 0.5).astype(int)
    labeled, num_features = label(binary)

    digit_regions = []
    for i in range(1, num_features + 1):
        coords = np.where(labeled == i)
        if len(coords[0]) < min_area:
            continue

        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()

        padding = 2
        y_min = max(0, y_min - padding)
        y_max = min(image.shape[0], y_max + padding)
        x_min = max(0, x_min - padding)
        x_max = min(image.shape[1], x_max + padding)

        cropped = gray[y_min:y_max, x_min:x_max]
        digit_regions.append({
            'image': cropped,
            'center': (x_min + (x_max - x_min) / 2, y_min + (y_max - y_min) / 2)
        })

    digit_regions.sort(key=lambda r: r['center'][0])
    return digit_regions

def recognize_digit(cropped_digit, model, scaler):
    processed = preprocess_single_digit(cropped_digit)
    processed_scaled = scaler.transform([processed])
    prediction = model.predict(processed_scaled)[0]
    confidence = np.max(model.predict_proba(processed_scaled)[0])
    return prediction, confidence


def recognize_with_clustering(cropped_digit, kmeans_model, scaler, y_labels):
    processed = preprocess_single_digit(cropped_digit)
    processed_scaled = scaler.transform([processed])
    cluster = kmeans_model.predict(processed_scaled)[0]
    cluster_mask = (kmeans_model.labels_ == cluster)
    most_common_digit = np.bincount(y_labels[cluster_mask].astype(int)).argmax()
    return most_common_digit

from PIL import Image, ImageDraw, ImageFont

width, height = 200, 100
image_pil = Image.new('L', (width, height), color=0)
draw = ImageDraw.Draw(image_pil)

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 60)
except:
    font = ImageFont.load_default()

text = "42"
bbox = draw.textbbox((0, 0), text, font=font)
text_width = bbox[2] - bbox[0]
text_height = bbox[3] - bbox[1]
x = (width - text_width) // 2
y = (height - text_height) // 2
draw.text((x, y), text, fill=255, font=font)

test_image = np.array(image_pil) / 255.0

plt.figure(figsize=(8, 4))
plt.imshow(test_image, cmap='gray')
plt.title("Тестовая картинка: число 42")
plt.axis('off')
plt.show()

digits_regions = extract_digits_from_image(test_image)

print(f"\nНайдено цифр: {len(digits_regions)}")

recognized_number = ""
for i, region in enumerate(digits_regions):
    digit, conf = recognize_digit(region['image'], model, scaler)
    recognized_number += str(digit)
    print(f"Цифра {i + 1}: {digit} (уверенность: {conf:.4f})")

    plt.figure(figsize=(2, 2))
    plt.imshow(region['image'], cmap='gray')
    plt.title(f"→ {digit}")
    plt.axis('off')
    plt.show()

print("\n" + "=" * 40)
print(f"РЕЗУЛЬТАТ: {recognized_number}")
print("=" * 40)

kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
kmeans.fit(X_scaled)

clustered_number = ""
for i, region in enumerate(digits_regions):
    digit = recognize_with_clustering(region['image'], kmeans, scaler, y)
    clustered_number += str(digit)

print(f"\nКластеризация распознала: {clustered_number}")