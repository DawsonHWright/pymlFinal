import os
import cv2
import joblib
import random
import numpy as np
import matplotlib

# Prevent blocking plots
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# =========================
# Paths
# =========================
BASE_DATASET_DIR = "dataset/handwritten-english-characters-and-digits/combined_folder"
TRAIN_DIR = os.path.join(BASE_DATASET_DIR, "train")
TEST_DIR = os.path.join(BASE_DATASET_DIR, "test")

EXTRA_DIGITS_DIR = "dataset/handwritten-digits-0-9"

MODEL_DIR = "models"
OUTPUT_DIR = "outputs"
MODEL_PATH = os.path.join(MODEL_DIR, "alphabetic_vs_numeric_rf.joblib")

# =========================
# Settings
# =========================
IMG_SIZE = 64
RANDOM_STATE = 42
TARGET_NUMERIC_TRAIN_COUNT = 2200

# =========================
# Setup
# =========================
def ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Label helper
# =========================
def get_binary_label(folder_name):
    if folder_name.isdigit():
        return 1  # numeric
    return 0      # alphabetic

# =========================
# Feature extraction
# =========================
def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read {image_path}")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0

    pixels = img.flatten()
    mean = np.mean(img)
    std = np.std(img)
    horiz = np.sum(img, axis=1)
    vert = np.sum(img, axis=0)

    return np.concatenate([pixels, [mean, std], horiz, vert])

# =========================
# Data collection
# =========================
def collect_original_data(folder):
    items = []
    for sub in os.listdir(folder):
        sub_path = os.path.join(folder, sub)
        if not os.path.isdir(sub_path):
            continue

        label = get_binary_label(sub)

        for file in os.listdir(sub_path):
            path = os.path.join(sub_path, file)
            if os.path.isfile(path):
                items.append((path, label))

    return items


def collect_extra_digits(folder):
    paths = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                paths.append(os.path.join(root, file))
    return paths

# =========================
# Build dataset
# =========================
def build_dataset(items):
    X, y = [], []

    for path, label in items:
        try:
            X.append(extract_features(path))
            y.append(label)
        except:
            continue

    return np.array(X), np.array(y)

# =========================
# Load training with balancing
# =========================
def load_train():
    original = collect_original_data(TRAIN_DIR)

    alpha = [(p, l) for p, l in original if l == 0]
    numeric = [(p, l) for p, l in original if l == 1]

    print(f"Original alphabetic: {len(alpha)}")
    print(f"Original numeric:    {len(numeric)}")

    needed = max(0, TARGET_NUMERIC_TRAIN_COUNT - len(numeric))

    extra_paths = collect_extra_digits(EXTRA_DIGITS_DIR)
    random.shuffle(extra_paths)

    extra = [(p, 1) for p in extra_paths[:needed]]

    print(f"Extra numeric added: {len(extra)}")

    combined = alpha + numeric + extra
    random.shuffle(combined)

    return build_dataset(combined)

# =========================
# Load test
# =========================
def load_test():
    items = collect_original_data(TEST_DIR)
    return build_dataset(items)

# =========================
# Main
# =========================
def main():
    ensure_dirs()

    print("Loading training data...")
    X_train, y_train = load_train()

    print("\nLoading test data...")
    X_test, y_test = load_test()

    print(f"\nTrain size: {len(y_train)}")
    print(f"Test size:  {len(y_test)}")

    print("\nTraining KNN model...")
    model = KNeighborsClassifier(
        n_neighbors=5
    )
    model.fit(X_train, y_train)

    print("\nEvaluating...")
    y_pred = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["alphabetic", "numeric"]))

    # =========================
    # Confusion Matrix
    # =========================
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["alphabetic", "numeric"])
    disp.plot()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plt.close()

    # =========================
    # Feature Importance
    # =========================
    importances = model.feature_importances_
    top_k = 20
    idx = np.argsort(importances)[-top_k:][::-1]

    plt.figure(figsize=(10, 5))
    plt.bar(range(top_k), importances[idx])
    plt.xticks(range(top_k), idx, rotation=45)
    plt.title("Top 20 Feature Importances")
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(save_path)
    print(f"Saved feature importance plot to: {save_path}")
    plt.close()

    print("\nDone.")

# =========================
if __name__ == "__main__":
    main()