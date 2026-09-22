import pandas as pd

# Load the dataset
df = pd.read_csv('/content/powerconsumption.csv')

# Display the first 5 rows of the DataFrame
print("First 5 rows of the dataset:")
display(df.head())

# Display information about the DataFrame, including data types and non-null values
print("\nDataFrame Info:")
df.info()

# Display descriptive statistics of the numerical columns
print("\nDescriptive Statistics:")
display(df.describe())

import matplotlib.pyplot as plt
import seaborn as sns

# Convert 'Datetime' column to datetime objects and set as index
df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.set_index('Datetime')

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Visualize Power Consumption for each zone
plt.figure(figsize=(18, 10))

plt.subplot(3, 1, 1)
plt.plot(df['PowerConsumption_Zone1'], label='Zone 1')
plt.title('Power Consumption - Zone 1')
plt.xlabel('Date')
plt.ylabel('Power Consumption')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(df['PowerConsumption_Zone2'], label='Zone 2', color='orange')
plt.title('Power Consumption - Zone 2')
plt.xlabel('Date')
plt.ylabel('Power Consumption')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(df['PowerConsumption_Zone3'], label='Zone 3', color='green')
plt.title('Power Consumption - Zone 3')
plt.xlabel('Date')
plt.ylabel('Power Consumption')
plt.legend()

plt.tight_layout()
plt.show()

from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Select features for the LSTM autoencoder (all power consumption zones)
features = ['PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3']
data = df[features].values

# Scale the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Define sequence length
TIME_STEPS = 30 # For example, use the last 30 time steps to predict the next

# Function to create sequences
def create_sequences(data, time_steps):
    X = []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
    return np.array(X)

# Create sequences for the scaled data
X_sequences = create_sequences(data_scaled, TIME_STEPS)

print(f"Original data shape: {data.shape}")
print(f"Scaled data shape: {data_scaled.shape}")
print(f"Shape of sequences for LSTM: {X_sequences.shape}")
print(f"First sequence example:\n{X_sequences[0]}")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense

# Define the input shape (TIME_STEPS, number of features)
INPUT_SHAPE = (X_sequences.shape[1], X_sequences.shape[2])

# Build the LSTM Autoencoder model
model = Sequential([
    # Encoder
    LSTM(units=128, activation='relu', input_shape=INPUT_SHAPE, return_sequences=False),
    RepeatVector(n=INPUT_SHAPE[0]), # Repeat the last output of the encoder for the decoder
    # Decoder
    LSTM(units=128, activation='relu', return_sequences=True),
    TimeDistributed(Dense(units=INPUT_SHAPE[1])) # Output a Dense layer for each time step
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Display model summary
print("LSTM Autoencoder Model Summary:")
model.summary()

from tensorflow.keras.callbacks import EarlyStopping

# Define callbacks for early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True)

# Train the model
history = model.fit(
    X_sequences, X_sequences, # Autoencoder: input and output are the same
    epochs=20, # You can adjust this
    batch_size=128, # You can adjust this
    validation_split=0.2, # Use 20% of data for validation
    callbacks=[early_stopping], # Add early stopping callback
    shuffle=False # Maintain sequence order
)

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss (Mean Squared Error)')
plt.legend()
plt.grid(True)
plt.show()

from sklearn.metrics import mean_absolute_error

# Get predictions (reconstructions) from the trained model
X_pred = model.predict(X_sequences)

# Calculate reconstruction errors for each sequence
# We will calculate MAE between original and reconstructed sequences
reconstruction_errors = np.array([mean_absolute_error(X_sequences[i], X_pred[i]) for i in range(len(X_sequences))])

# Extend the reconstruction errors to match the original dataframe's length for plotting
# The first TIME_STEPS elements don't have a full sequence for error calculation,
# so we pad them with zeros or NaNs. Here, we'll use a simplified approach for visualization.
# The errors correspond to the *end* of each sequence.
full_reconstruction_errors = np.full(data_scaled.shape[0], np.nan)
full_reconstruction_errors[TIME_STEPS:] = reconstruction_errors

# Plot reconstruction errors
plt.figure(figsize=(15, 7))
plt.plot(df.index, full_reconstruction_errors, label='Reconstruction Error')
plt.title('Reconstruction Error Over Time')
plt.xlabel('Time')
plt.ylabel('Reconstruction Error (MAE)')
plt.legend()
plt.grid(True)
plt.show()

import numpy as np

# Get predictions (reconstructions) from the trained model
# X_pred = model.predict(X_sequences) # This line should be before this cell, as it was in the original flow

# Calculate reconstruction errors for each sequence
# We will calculate MAE between original and reconstructed sequences
# reconstruction_errors = np.array([mean_absolute_error(X_sequences[i], X_pred[i]) for i in range(len(X_sequences))]) # This line should be before this cell

# Calculate statistics of reconstruction errors
mean_error = np.mean(reconstruction_errors)
std_error = np.std(reconstruction_errors)

# Determine a threshold for anomalies
# A common approach is to use a multiple of the standard deviation
threshold = mean_error + 2 * std_error # 2 standard deviations above the mean

print(f"\n--- Anomaly Detection Metrics ---")
print(f"Mean Reconstruction Error (MAE): {mean_error:.4f}")
print(f"Standard Deviation of Reconstruction Error: {std_error:.4f}")
print(f"Calculated Anomaly Threshold (Mean + 2*Std): {threshold:.4f}")

# Identify anomalies
anomalies = np.where(reconstruction_errors > threshold)[0]

# Map anomalies back to original DataFrame indices
# Note: anomalies are identified on sequences, so their timestamp corresponds to the end of the sequence
anomaly_dates = df.index[TIME_STEPS + anomalies]

print(f"Number of detected anomalies: {len(anomalies)}")
print("Dates of first 10 detected anomalies (if any):")
if len(anomaly_dates) > 0:
    for date in anomaly_dates[:10]: # Print first 10 anomalies if many
        print(date)
    if len(anomaly_dates) > 10:
        print("...")
else:
    print("No anomalies detected above the threshold.")

import matplotlib.dates as mdates

# Create a DataFrame for plotting anomalies
anomaly_df = pd.DataFrame(df.iloc[TIME_STEPS:].index)
anomaly_df['reconstruction_error'] = reconstruction_errors
anomaly_df['anomaly'] = anomaly_df['reconstruction_error'] > threshold

# Merge anomaly information back to the original DataFrame
plot_df = df.copy()
plot_df = plot_df.iloc[TIME_STEPS:] # Align data for plotting errors
plot_df['reconstruction_error'] = reconstruction_errors
plot_df['anomaly'] = plot_df['reconstruction_error'] > threshold

# Visualize anomalies for each power consumption zone
plt.figure(figsize=(20, 15))

for i, feature in enumerate(features):
    plt.subplot(len(features), 1, i + 1)
    plt.plot(plot_df.index, plot_df[feature], label=f'{feature} (Normal)')

    # Plot anomalies
    anomalous_data = plot_df[plot_df['anomaly']]
    if not anomalous_data.empty:
        plt.scatter(
            anomalous_data.index,
            anomalous_data[feature],
            color='red',
            label='Anomaly',
            s=20,
            marker='o'
        )

    plt.title(f'Power Consumption - {feature} with Anomalies')
    plt.xlabel('Date')
    plt.ylabel('Power Consumption')
    plt.legend()
    plt.grid(True)

    # Format x-axis for better date display
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate() # Auto-format date labels

plt.tight_layout()
plt.show()
