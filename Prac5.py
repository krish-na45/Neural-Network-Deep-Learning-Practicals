# ============================================================
# FLOWER IMAGE CLASSIFICATION USING TRANSFER LEARNING
# MobileNetV2 Pre-trained on ImageNet
# Google Colab - Complete Code
# ============================================================

# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

print("TensorFlow Version:", tf.__version__)


# ------------------------------------------------------------
# 2. DOWNLOAD FLOWER DATASET
# ------------------------------------------------------------
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

data_dir = tf.keras.utils.get_file(
    "flower_photos",
    origin=dataset_url,
    untar=True
)

# Correct data_dir to point to the directory containing actual class subfolders
# The dataset usually unpacks into a nested 'flower_photos' directory
data_dir = os.path.join(data_dir, 'flower_photos')

print("\nDataset downloaded successfully!")
print("Dataset path:", data_dir)


# ------------------------------------------------------------
# 3. PARAMETERS
# ------------------------------------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 123


# ------------------------------------------------------------
# 4. LOAD DATASET
# ------------------------------------------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

validation_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("\nFlower Classes:")
for i, name in enumerate(class_names):
    print(i, ":", name)

print("\nNumber of classes:", num_classes)


# ------------------------------------------------------------
# 5. DISPLAY SAMPLE IMAGES
# ------------------------------------------------------------
plt.figure(figsize=(10, 8))

for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

plt.suptitle("Sample Flower Images", fontsize=16)
plt.show()


# ------------------------------------------------------------
# 6. IMPROVE DATA PIPELINE PERFORMANCE
# ------------------------------------------------------------
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.prefetch(buffer_size=AUTOTUNE)


# ------------------------------------------------------------
# 7. DATA AUGMENTATION
# ------------------------------------------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.1)
])


# ------------------------------------------------------------
# 8. LOAD PRE-TRAINED MOBILENETV2
# ------------------------------------------------------------
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Initially freeze all pretrained layers
base_model.trainable = False

print("\nMobileNetV2 loaded with ImageNet weights.")
print("Pre-trained layers are initially frozen.")


# ------------------------------------------------------------
# 9. CREATE TRANSFER LEARNING MODEL
# ------------------------------------------------------------
inputs = layers.Input(shape=(224, 224, 3))

# Data augmentation
x = data_augmentation(inputs)

# MobileNetV2 preprocessing
x = preprocess_input(x)

# Extract features using MobileNetV2
x = base_model(x, training=False)

# Convert feature maps to a single vector
x = layers.GlobalAveragePooling2D()(x)

# Dropout to reduce overfitting
x = layers.Dropout(0.2)(x)

# Flower classification layer
outputs = layers.Dense(
    num_classes,
    activation="softmax"
)(x)

model = models.Model(inputs, outputs)


# ------------------------------------------------------------
# 10. COMPILE MODEL
# ------------------------------------------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel created successfully!")


# ------------------------------------------------------------
# 11. DISPLAY MODEL SUMMARY
# ------------------------------------------------------------
model.summary()


# ------------------------------------------------------------
# 12. CALLBACKS
# ------------------------------------------------------------
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    min_lr=1e-7
)


# ------------------------------------------------------------
# 13. TRAIN TRANSFER LEARNING MODEL
# ------------------------------------------------------------
INITIAL_EPOCHS = 10

print("\n==========================================")
print("STARTING TRANSFER LEARNING")
print("==========================================")

history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=INITIAL_EPOCHS,
    callbacks=[early_stopping, reduce_lr]
)


# ------------------------------------------------------------
# 14. FINE-TUNE MOBILENETV2
# ------------------------------------------------------------
print("\n==========================================")
print("STARTING FINE-TUNING")
print("==========================================")

# Unfreeze MobileNetV2
base_model.trainable = True

# Freeze the first 100 layers
for layer in base_model.layers[:100]:
    layer.trainable = False

# Recompile using a very small learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

FINE_TUNE_EPOCHS = 10

total_epochs = INITIAL_EPOCHS + FINE_TUNE_EPOCHS

history_fine = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=total_epochs,
    initial_epoch=len(history.history["loss"]),
    callbacks=[early_stopping, reduce_lr]
)


# ------------------------------------------------------------
# 15. COMBINE TRAINING HISTORY
# ------------------------------------------------------------
train_accuracy = (
    history.history["accuracy"] +
    history_fine.history["accuracy"]
)

val_accuracy = (
    history.history["val_accuracy"] +
    history_fine.history["val_accuracy"]
)

train_loss = (
    history.history["loss"] +
    history_fine.history["loss"]
)

val_loss = (
    history.history["val_loss"] +
    history_fine.history["val_loss"]
)


# ------------------------------------------------------------
# 16. EVALUATE MODEL
# ------------------------------------------------------------
print("\n==========================================")
print("FINAL MODEL EVALUATION")
print("==========================================")

test_loss, test_accuracy = model.evaluate(
    validation_ds,
    verbose=1
)

print("\n------------------------------------------")
print("FINAL RESULTS")
print("------------------------------------------")
print("Validation Loss     :", round(test_loss, 4))
print("Validation Accuracy :", round(test_accuracy * 100, 2), "%")
print("------------------------------------------")


# ------------------------------------------------------------
# 17. DISPLAY ACCURACY
# ------------------------------------------------------------
final_train_accuracy = train_accuracy[-1] * 100
final_val_accuracy = val_accuracy[-1] * 100

print("\n==========================================")
print("ACCURACY")
print("==========================================")
print(f"Training Accuracy   : {final_train_accuracy:.2f}%")
print(f"Validation Accuracy : {final_val_accuracy:.2f}%")
print(f"Final Model Accuracy: {test_accuracy * 100:.2f}%")
print("==========================================")


# ------------------------------------------------------------
# 18. PLOT ACCURACY AND LOSS
# ------------------------------------------------------------
plt.figure(figsize=(14, 5))

# Accuracy plot
plt.subplot(1, 2, 1)

plt.plot(
    train_accuracy,
    label="Training Accuracy",
    linewidth=2
)

plt.plot(
    val_accuracy,
    label="Validation Accuracy",
    linewidth=2
)

plt.axvline(
    x=INITIAL_EPOCHS - 1,
    color="red",
    linestyle="--",
    label="Fine-Tuning Begins"
)

plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)


# Loss plot
plt.subplot(1, 2, 2)

plt.plot(
    train_loss,
    label="Training Loss",
    linewidth=2
)

plt.plot(
    val_loss,
    label="Validation Loss",
    linewidth=2
)

plt.axvline(
    x=INITIAL_EPOCHS - 1,
    color="red",
    linestyle="--",
    label="Fine-Tuning Begins"
)

plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 19. PREDICT SOME VALIDATION IMAGES
# ------------------------------------------------------------
print("\n==========================================")
print("SAMPLE PREDICTIONS")
print("==========================================")

plt.figure(figsize=(12, 10))

for images, labels in validation_ds.take(1):

    predictions = model.predict(images, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)

    for i in range(min(9, len(images))):

        ax = plt.subplot(3, 3, i + 1)

        plt.imshow(images[i].numpy().astype("uint8"))

        actual = class_names[labels[i]]
        predicted = class_names[predicted_labels[i]]
        confidence = np.max(predictions[i]) * 100

        if actual == predicted:
            color = "green"
        else:
            color = "red"

        plt.title(
            f"Actual: {actual}\n"
            f"Predicted: {predicted}\n"
            f"Confidence: {confidence:.1f}%",
            color=color
        )

        plt.axis("off")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 20. SAVE THE TRAINED MODEL
# ------------------------------------------------------------
model.save("flower_mobilenetv2_transfer_learning.keras")

print("\nModel saved as:")
print("flower_mobilenetv2_transfer_learning.keras")
