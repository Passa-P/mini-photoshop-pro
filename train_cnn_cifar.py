import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

# =====================================================================
# KONFIGURASI
# =====================================================================
# Kelas yang dipakai dari CIFAR-10
# CIFAR-10 ID → nama kelas kita
TARGET_MAP = {
    1: "automobile",
    3: "cat",
    5: "dog",
    7: "horse"
}

CLASS_NAMES  = list(TARGET_MAP.values())   # ['automobile', 'cat', 'dog', 'horse']
NUM_CLASSES  = len(CLASS_NAMES)            # 4
IMG_SIZE     = (224, 224)                  # input MobileNetV2
BATCH_SIZE   = 16                          # kecil agar aman di CPU
EPOCHS_PHASE1 = 10                         # training top layer
EPOCHS_PHASE2 = 20                         # fine-tuning
MODEL_OUTPUT  = "model_custom.h5"

print("=" * 55)
print("  Mini Photoshop Pro — CNN Training (CIFAR-10)")
print("=" * 55)
print(f"Kelas: {CLASS_NAMES}")
print(f"Jumlah kelas: {NUM_CLASSES}")
print()

# =====================================================================
# STEP 1: LOAD & FILTER DATASET CIFAR-10
# =====================================================================
print("📥 Mengunduh CIFAR-10 (otomatis)...")
(x_train_all, y_train_all), (x_test_all, y_test_all) = \
    tf.keras.datasets.cifar10.load_data()

y_train_all = y_train_all.flatten()
y_test_all  = y_test_all.flatten()

def filter_classes(x, y, target_map):
    """Ambil hanya kelas yang ada di target_map, remap label ke 0,1,2,..."""
    original_ids = list(target_map.keys())
    mask = np.isin(y, original_ids)
    x_filtered = x[mask]
    y_filtered = y[mask]

    # Remap: misal {1→0, 3→1, 5→2, 7→3}
    remap = {orig: new for new, orig in enumerate(original_ids)}
    y_remapped = np.array([remap[label] for label in y_filtered])

    return x_filtered, y_remapped

x_train, y_train = filter_classes(x_train_all, y_train_all, TARGET_MAP)
x_val,   y_val   = filter_classes(x_test_all,  y_test_all,  TARGET_MAP)

print(f"✅ Data training : {len(x_train)} foto")
print(f"✅ Data validasi : {len(x_val)} foto")
print()

# Distribusi per kelas
for i, name in enumerate(CLASS_NAMES):
    count_train = np.sum(y_train == i)
    count_val   = np.sum(y_val == i)
    print(f"   {name:12s} → train: {count_train}, val: {count_val}")
print()

# =====================================================================
# STEP 2: PREPROCESSING & AUGMENTASI
# =====================================================================
print("🔄 Memproses gambar...")

def preprocess(x, y, augment=False):
    """
    CIFAR-10 aslinya 32x32 — resize ke 224x224 untuk MobileNetV2.
    Normalisasi ke [0, 1].
    """
    dataset = tf.data.Dataset.from_tensor_slices((x, y))

    def resize_and_normalize(img, label):
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.image.resize(img, IMG_SIZE)
        return img, tf.one_hot(label, NUM_CLASSES)

    def augment_fn(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.15)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        img = tf.image.random_saturation(img, 0.8, 1.2)
        img = tf.clip_by_value(img, 0.0, 1.0)
        return img, label

    dataset = dataset.map(resize_and_normalize,
                          num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        dataset = dataset.map(augment_fn,
                              num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset

train_ds = preprocess(x_train, y_train, augment=True)
val_ds   = preprocess(x_val,   y_val,   augment=False)

print("✅ Preprocessing selesai.")
print()

# =====================================================================
# STEP 3: BANGUN MODEL (MobileNetV2 + Custom Head)
# =====================================================================
print("🏗️  Membangun model CNN (MobileNetV2 + Transfer Learning)...")

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze dulu di phase 1

x      = base_model.output
x      = GlobalAveragePooling2D()(x)
x      = Dropout(0.3)(x)
x      = Dense(256, activation='relu')(x)
x      = Dropout(0.2)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"✅ Model siap. Total parameter: {model.count_params():,}")
print()

# =====================================================================
# STEP 4: PHASE 1 — TRAIN TOP LAYERS
# =====================================================================
print("=" * 55)
print("  PHASE 1: Training top layers (base model frozen)")
print("=" * 55)

callbacks_p1 = [
    ModelCheckpoint(
        MODEL_OUTPUT,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=4,
        restore_best_weights=True,
        verbose=1
    )
]

history1 = model.fit(
    train_ds,
    epochs=EPOCHS_PHASE1,
    validation_data=val_ds,
    callbacks=callbacks_p1,
    verbose=1
)

print()
print(f"✅ Phase 1 selesai. Best val_accuracy: "
      f"{max(history1.history['val_accuracy']):.4f}")
print()

# =====================================================================
# STEP 5: PHASE 2 — FINE-TUNING
# =====================================================================
print("=" * 55)
print("  PHASE 2: Fine-tuning (unfreeze 30 layer terakhir)")
print("=" * 55)

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Hitung layer yang ditraining
trainable_count = sum(1 for l in model.layers if l.trainable)
print(f"Layer yang ditraining: {trainable_count}")
print()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_p2 = [
    ModelCheckpoint(
        MODEL_OUTPUT,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=6,
        restore_best_weights=True,
        verbose=1
    )
]

history2 = model.fit(
    train_ds,
    epochs=EPOCHS_PHASE2,
    validation_data=val_ds,
    callbacks=callbacks_p2,
    verbose=1
)

print()
print(f"✅ Phase 2 selesai. Best val_accuracy: "
      f"{max(history2.history['val_accuracy']):.4f}")
print()

# =====================================================================
# STEP 6: SIMPAN MODEL & CLASS NAMES
# =====================================================================
model.save(MODEL_OUTPUT)

# Simpan nama kelas ke file teks agar bisa dibaca app
with open("class_names.txt", "w") as f:
    for name in CLASS_NAMES:
        f.write(name + "\n")

print(f"✅ Model disimpan   : {MODEL_OUTPUT}")
print(f"✅ Kelas disimpan   : class_names.txt")
print(f"   Isi kelas        : {CLASS_NAMES}")
print()

# =====================================================================
# STEP 7: PLOT GRAFIK TRAINING
# =====================================================================
acc      = history1.history['accuracy']     + history2.history['accuracy']
val_acc  = history1.history['val_accuracy'] + history2.history['val_accuracy']
loss     = history1.history['loss']         + history2.history['loss']
val_loss = history1.history['val_loss']     + history2.history['val_loss']
epochs_range = range(1, len(acc) + 1)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle("Hasil Training CNN — CIFAR-10 (MobileNetV2)",
             fontsize=13, fontweight='bold')

# Accuracy
axes[0].plot(epochs_range, acc,     label='Train Accuracy', color='#D4AF37')
axes[0].plot(epochs_range, val_acc, label='Val Accuracy',   color='#4A90D9')
axes[0].axvline(x=len(history1.history['accuracy']),
                color='red', linestyle='--', alpha=0.5, label='Phase 1→2')
axes[0].set_title('Accuracy'); axes[0].legend(); axes[0].set_xlabel('Epoch')

# Loss
axes[1].plot(epochs_range, loss,     label='Train Loss', color='#D4AF37')
axes[1].plot(epochs_range, val_loss, label='Val Loss',   color='#4A90D9')
axes[1].axvline(x=len(history1.history['loss']),
                color='red', linestyle='--', alpha=0.5, label='Phase 1→2')
axes[1].set_title('Loss'); axes[1].legend(); axes[1].set_xlabel('Epoch')

plt.tight_layout()
plt.savefig("training_result.png", dpi=120, bbox_inches='tight')
plt.show()

print("📊 Grafik disimpan : training_result.png")
print()
print("=" * 55)
print("  TRAINING SELESAI!")
print(f"  Akurasi terbaik  : {max(val_acc):.2%}")
print(f"  Jalankan app     : python mini_photoshop_pro.py")
print("=" * 55)