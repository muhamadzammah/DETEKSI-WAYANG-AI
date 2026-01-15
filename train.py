import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns

# ===============================
# CONFIG (HASIL TUNING)
# ===============================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
VAL_SPLIT = 0.2

# Hyperparameter hasil tuning
LEARNING_RATE = 1e-4
DENSE_1 = 512
DENSE_2 = 256
DROPOUT_1 = 0.4
DROPOUT_2 = 0.3
UNFREEZE_LAYERS = 20

DATASET_DIR = "dataset_wayang"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# ===============================
# DATA GENERATOR (OPTIMAL)
# ===============================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=VAL_SPLIT,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False  # WAYANG TIDAK SIMETRIS
)

train_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

NUM_CLASSES = train_data.num_classes
CLASS_NAMES = list(train_data.class_indices.keys())

print("Jumlah kelas:", NUM_CLASSES)
print("Nama kelas:", CLASS_NAMES)

# ===============================
# CLASS WEIGHT (WAJIB)
# ===============================
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weights = dict(enumerate(class_weights))

print("Class weight digunakan")

# ===============================
# MODEL: MobileNetV2 (TUNED)
# ===============================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze semua layer
for layer in base_model.layers:
    layer.trainable = False

# Unfreeze layer terakhir
for layer in base_model.layers[-UNFREEZE_LAYERS:]:
    layer.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)

x = Dense(DENSE_1, activation="relu")(x)
x = Dropout(DROPOUT_1)(x)

x = Dense(DENSE_2, activation="relu")(x)
x = Dropout(DROPOUT_2)(x)

output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ===============================
# CALLBACKS (AGRESIF TAPI AMAN)
# ===============================
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        patience=2,
        factor=0.3,
        min_lr=1e-6,
        verbose=1
    )
]

# ===============================
# TRAINING
# ===============================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

# ===============================
# SAVE MODEL
# ===============================
model_path = os.path.join(MODEL_DIR, "wayang_model.h5")
model.save(model_path)
print(f"\n✅ Model berhasil disimpan di: {model_path}")

# ===============================
# EVALUATION
# ===============================
print("\n📊 Evaluasi Model (VALIDATION DATA)")

y_true = val_data.classes
y_pred_prob = model.predict(val_data)
y_pred = np.argmax(y_pred_prob, axis=1)

print("\n📄 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(14, 12))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - MobileNetV2 Wayang")
plt.tight_layout()
plt.show()
