import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = mnist.load_data()

print(f"Training data shape: {X_train_raw.shape}")
print(f"Training labels shape: {y_train_raw.shape}")
print(f"Test data shape: {X_test_raw.shape}")
print(f"Test labels shape: {y_test_raw.shape}")

X_train = X_train_raw.astype('float32') / 255.0
X_test = X_test_raw.astype('float32') / 255.0

y_train = to_categorical(y_train_raw, 10)
y_test = to_categorical(y_test_raw, 10)

print(f"Normalized training data shape: {X_train.shape}")
print(f"One-hot encoded training labels shape: {y_train.shape}")

model = Sequential([
    Flatten(input_shape=(28, 28)), # Input layer: Flattens the 28x28 images into a 1D vector of 784 pixels
    Dense(128, activation='relu'), # First hidden layer with 128 neurons and ReLU activation
    Dense(64, activation='relu'),  # Second hidden layer with 64 neurons and ReLU activation
    Dense(10, activation='softmax') # Output layer with 10 neurons (for 10 classes) and Softmax activation
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    X_train,
    y_train,
    epochs=10, # Number of times the model will go through the entire training dataset
    batch_size=32, # Number of samples per gradient update
    validation_split=0.1 # Use 10% of training data for validation during training
)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")



# Get the model's predictions on the test set
predictions = model.predict(X_test)

# Convert one-hot encoded y_test back to original labels for comparison
y_test_labels = np.argmax(y_test, axis=1)

# Select a few images to display
num_images_to_display = 10

plt.figure(figsize=(15, 6))
for i in range(num_images_to_display):
    # Display the image
    plt.subplot(2, 5, i + 1) # 2 rows, 5 columns
    plt.imshow(X_test_raw[i], cmap='gray')
    
    # Get actual and predicted labels
    actual_label = y_test_labels[i]
    predicted_label = np.argmax(predictions[i])
    
    # Set title based on prediction correctness
    color = 'green' if predicted_label == actual_label else 'red'
    plt.title(f'Actual: {actual_label}\nPred: {predicted_label}', color=color)
    plt.axis('off')

plt.tight_layout()
plt.show()
