import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import joblib

# =============================
# Konfigurasi dan Parameter
# =============================
DATA_DIR = 'data'  # Folder dataset
CATEGORIES = ['plastic', 'paper']  # Label kelas
IMG_SIZE = 64  # Ukuran gambar setelah resize (64x64)

# =============================
# Fungsi untuk Load dan Ekstraksi Fitur Gambar
# =============================
def load_data():
    X, y = [], []
    for label, category in enumerate(CATEGORIES):
        folder = os.path.join(DATA_DIR, category)
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                img = cv2.imread(file_path)  # Baca gambar
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Konversi ke RGB
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize
                img = img / 255.0  # Normalisasi ke [0,1]
                X.append(img.flatten())  # Flatten ke 1D array
                y.append(label)  # Label: 0=plastic, 1=paper
            except Exception as e:
                print(f"Gagal memproses {file_path}: {e}")
    return np.array(X), np.array(y)

print('Memuat data...')
X, y = load_data()
print(f'Jumlah data: {len(X)}')

# =============================
# Split Data: Training & Testing
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# =============================
# Training Model Random Forest
# =============================
print('Training Random Forest...')
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# =============================
# Evaluasi Model
# =============================
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)  # Akurasi
cm = confusion_matrix(y_test, y_pred)  # Confusion matrix
cr = classification_report(y_test, y_pred, target_names=CATEGORIES)  # Laporan klasifikasi

print(f'Akurasi: {acc:.4f}')
print('Confusion Matrix:')
print(cm)
print('Classification Report:')
print(cr)

# =============================
# Simpan Model ke File
# =============================
joblib.dump(clf, 'model.pkl')
print('Model disimpan ke model.pkl')

# =============================
# Visualisasi Confusion Matrix
# =============================
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CATEGORIES, yticklabels=CATEGORIES)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('static/confusion_matrix.png')
print('Visualisasi confusion matrix disimpan ke static/confusion_matrix.png')

# =============================
# Visualisasi Feature Importance
# =============================
importances = clf.feature_importances_  # Nilai pentingnya setiap fitur (pixel)
# Ambil 20 fitur teratas (karena fitur = pixel, visualisasi semua tidak informatif)
indices = np.argsort(importances)[-20:][::-1]
plt.figure(figsize=(10,6))
sns.barplot(x=importances[indices], y=[f'Pixel {i}' for i in indices])
plt.title('20 Feature Importances Teratas (Pixel)')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('static/feature_importance.png')
print('Visualisasi feature importance disimpan ke static/feature_importance.png')
