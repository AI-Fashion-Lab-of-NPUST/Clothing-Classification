import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\admin\Desktop\Clothing_Classifier"

DATASET_DIR = os.path.join(
    BASE_DIR,
    "filtered_5_classes",
    "dataset2"
)

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "training_results_v6"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

INITIAL_EPOCHS = 30
FINE_TUNE_EPOCHS = 30

SEED = 42


# ============================================================
# GPU CHECK
# ============================================================

print("=" * 65)
print("CLOTHING CLASSIFIER V6")
print("=" * 65)

print("TensorFlow:", tf.__version__)

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print("GPU detected:", gpus)
else:
    print("WARNING: No GPU detected. Training will use CPU.")


# ============================================================
# LOAD DATASETS
# ============================================================

print("\nLoading datasets...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names

NUM_CLASSES = len(class_names)

print("\nClasses:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")

print("\nNumber of classes:", NUM_CLASSES)


# ============================================================
# SAVE CLASS NAMES
# ============================================================

with open(
    os.path.join(OUTPUT_DIR, "class_names.json"),
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        class_names,
        f,
        indent=4
    )


# ============================================================
# PERFORMANCE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.15),
    ],
    name="data_augmentation"
)


# ============================================================
# BASE MODEL
# ============================================================

print("\nCreating EfficientNetB0...")

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

# Freeze backbone initially
base_model.trainable = False


# ============================================================
# CLASSIFIER
# ============================================================

inputs = layers.Input(
    shape=(224, 224, 3)
)

x = data_augmentation(inputs)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.30)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

model = models.Model(
    inputs,
    outputs
)


# ============================================================
# INITIAL TRAINING
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


BEST_MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "best_clothing_model_v2.keras"
)


callbacks_initial = [
    # حفظ أفضل نموذج بناءً على دقة التحقق (وهو هدفك النهائي)
    ModelCheckpoint(
        BEST_MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    # تقليل معدل التعلم بسرعة إذا ثبتت الخسارة (لأن التدريب المبدئي سريع)
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,            # تقليل بمقدار الخُمس (مثلاً من 1e-3 إلى 2e-4) مناسب جداً هنا
        patience=2,            # صبر قصير (حقبتين فقط) لأن الطبقات العلوية تتقارب بسرعة
        min_lr=1e-6,
        verbose=1
    ),

    # إيقاف التدريب إذا لم تتحسن الخسارة (صبر متناسق مع تقليل معدل التعلم)
    EarlyStopping(
        monitor="val_loss",    # نراقب الخسارة لضمان استقرار المنحنى
        patience=6,            # يعطي فرصة لـ 4 حقب كاملة بعد أول تقليل لمعدل التعلم
        restore_best_weights=True,
        mode="min",
        verbose=1
    )
]


print("\n")
print("=" * 65)
print("PHASE 1 — INITIAL TRAINING")
print("=" * 65)

history_initial = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    callbacks=callbacks_initial
)


# ============================================================
# FINE-TUNING
# ============================================================

print("\n")
print("=" * 65)
print("PHASE 2 — FINE-TUNING")
print("=" * 65)

# Unfreeze backbone
base_model.trainable = True

# Freeze first 70% of layers
freeze_until = int(
    len(base_model.layers) * 0.70
)

for layer in base_model.layers[:freeze_until]:
    layer.trainable = False

# Keep BatchNorm layers frozen
for layer in base_model.layers:

    if isinstance(
        layer,
        layers.BatchNormalization
    ):
        layer.trainable = False


print(
    f"Total backbone layers: "
    f"{len(base_model.layers)}"
)

print(
    f"Frozen backbone layers: "
    f"{freeze_until}"
)

print(
    "Fine-tunable layers:",
    sum(
        1
        for layer in base_model.layers
        if layer.trainable
    )
)


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


callbacks_finetune = [
    # استمرار حفظ أفضل نموذج بناءً على أعلى دقة تحقق
    ModelCheckpoint(
        BEST_MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    # تقليل معدل التعلم ببطء شديد وبحذر
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,            # تقليل هادئ (النصف فقط) للحفاظ على الميزات المدربة مسبقاً
        patience=4,            # الانتظار لـ 4 حقب للتأكد من أن النموذج يحتاج فعلاً للتقليل
        min_lr=1e-8,           # خفض الحد الأدنى لأن الـ Fine-tuning يتطلب أحياناً معدلات تعلم متناهية الصغر
        verbose=1
    ),

    # إيقاف التدريب بصبر طويل جداً لمنع الإنهاء المبكر
    EarlyStopping(
        monitor="val_loss",
        patience=12,           # يعطي النموذج فرصة للمرور بـ 3 دورات تقليل لمعدل التعلم (4 * 3 = 12)
        min_delta=0.0001,      # تحسين القيمة لـ 0.0001 لأن التغيرات في الـ Fine-Tuning تكون صغيرة جداً
        restore_best_weights=True,
        mode="min",
        verbose=1
    )
]


history_finetune = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=callbacks_finetune
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best validation model...")

best_model = tf.keras.models.load_model(
    BEST_MODEL_PATH
)


# ============================================================
# VALIDATION EVALUATION
# ============================================================

print("\n")
print("=" * 65)
print("VALIDATION RESULTS")
print("=" * 65)

val_loss, val_accuracy = best_model.evaluate(
    val_ds,
    verbose=1
)

print(
    f"\nValidation Loss: "
    f"{val_loss:.4f}"
)

print(
    f"Validation Accuracy: "
    f"{val_accuracy * 100:.2f}%"
)


# ============================================================
# INTERNAL TEST EVALUATION
# ============================================================

print("\n")
print("=" * 65)
print("INTERNAL TEST RESULTS")
print("=" * 65)

test_loss, test_accuracy = best_model.evaluate(
    test_ds,
    verbose=1
)

print(
    f"\nInternal Test Loss: "
    f"{test_loss:.4f}"
)

print(
    f"Internal Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

FINAL_MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "clothing_classifier_v6_final.keras"
)

best_model.save(
    FINAL_MODEL_PATH
)


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 65)
print("TRAINING COMPLETED")
print("=" * 65)

print(
    f"Best model:\n"
    f"{BEST_MODEL_PATH}"
)

print(
    f"\nFinal model:\n"
    f"{FINAL_MODEL_PATH}"
)

print(
    f"\nValidation accuracy: "
    f"{val_accuracy * 100:.2f}%"
)

print(
    f"Internal test accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print("=" * 65)