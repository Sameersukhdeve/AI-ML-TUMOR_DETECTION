# model/train_model.py  — FASTER VERSION

import os
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR   = os.path.join(BASE_DIR, 'dataset', 'Training')
TEST_DIR    = os.path.join(BASE_DIR, 'dataset', 'Testing')
MODEL_SAVE  = os.path.join(BASE_DIR, 'tumor_model.h5')

# ── Smaller image size = much faster ─────────────────────────
IMG_SIZE    = (128, 128)   # was 224x224 — this alone saves 70% time
BATCH_SIZE  = 16           # smaller batch = less RAM used
EPOCHS      = 10

# ── Data ──────────────────────────────────────────────────────
train_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
test_gen  = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

print("\nClasses:", train_data.class_indices)
NUM_CLASSES = len(train_data.class_indices)

# ── MobileNetV2 — tiny and fast, still accurate ───────────────
# VGG16 is huge (138M params) — too heavy for laptop CPU
# MobileNetV2 has only 3.4M params — runs 10x faster
base = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(128, 128, 3)
)

# Freeze base — only train our top layers
for layer in base.layers:
    layer.trainable = False

x      = base.output
x      = GlobalAveragePooling2D()(x)
x      = Dense(128, activation='relu')(x)
x      = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ── EarlyStopping — auto stops if no improvement ─────────────
# Saves time — no point running all 10 epochs if already good
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=3,          # stop if no improvement for 3 epochs
    restore_best_weights=True
)

print("\nTraining started — should finish in 20-40 minutes...\n")

history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=test_data,
    callbacks=[early_stop]
)

model.save(MODEL_SAVE)
print(f"\nDone! Model saved.")
print(f"Train accuracy : {round(history.history['accuracy'][-1]*100, 2)}%")
print(f"Val accuracy   : {round(history.history['val_accuracy'][-1]*100, 2)}%")