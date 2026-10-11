import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# 1. Load Dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# 2. Normalize Data
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# 3. Flatten Images
X_train = X_train.reshape(X_train.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)
print("Flattened training data shape:", X_train.shape)
print("Flattened testing data shape:", X_test.shape)

# 4. Define Input and Latent Dimension
input_dim = 784
encoding_dim = 32
input_layer = Input(shape=(input_dim,))

# 5. Build Encoder
encoder_layer = Dense(128, activation="relu")(input_layer)
encoder_layer = Dense(64, activation="relu")(encoder_layer)
encoded = Dense(encoding_dim, activation="relu")(encoder_layer)

# 6. Build Decoder
decoder_layer = Dense(64, activation="relu")(encoded)
decoder_layer = Dense(128, activation="relu")(decoder_layer)
decoded = Dense(input_dim, activation="sigmoid")(decoder_layer)

# 7. Create Autoencoder
autoencoder = Model(input_layer, decoded)

# 8. Create Encoder Model
encoder = Model(input_layer, encoded)

# 9. Compile Autoencoder
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy")

# 10. Train Autoencoder
history = autoencoder.fit(
    X_train, X_train,
    epochs=20,
    batch_size=256,
    shuffle=True,
    validation_data=(X_test, X_test)
)

# 11. Plot Training Loss
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Autoencoder Training Loss")
plt.legend()
plt.show()
