# train_cnn.py
# This is our most powerful model.
# 1. Deeper and Wider Conv1D layers to find B vs C.
# 2. Still uses EarlyStopping to prevent overfitting.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping

print("Starting FINAL training script (Deeper Conv1D Model)...")

# 1. Load the dataset
print("Loading data from keypoint.csv...")
data = pd.read_csv('keypoint.csv')

# 2. Separate features (X) and labels (y)
X = data.iloc[:, 1:]  # (42 keypoints)
y = data.iloc[:, 0]   # (gesture name)

# 3. Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(np.unique(y_encoded))
print(f"Found {num_classes} unique classes (gestures).")
y_categorical = to_categorical(y_encoded, num_classes=num_classes)

# 4. Split data (using a 30% test split)
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.3, random_state=42)
print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# 5. --- RESHAPE DATA FOR CONV1D ---
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)
print(f"Reshaped X_train shape: {X_train.shape}") # (samples, 42, 1)

# 6. --- Define the DEEPER Conv1D Model ---
print("Building the Deeper Conv1D model...")
model = Sequential()
model.add(Input(shape=(42, 1)))

# Block 1
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
model.add(BatchNormalization()) # Helps stabilize training
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))

# Block 2
model.add(Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))

# Block 3
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))

# Flatten the results
model.add(Flatten())

# The "brain"
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

# Final output layer
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 7. Train the model with Callbacks
print("Training the model with EarlyStopping...")

# Stop training if `val_loss` doesn't improve for 5 epochs
early_stopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1) # Increased patience

# Save the best model based on `val_loss`
checkpoint = ModelCheckpoint(
    'isl_cnn_model.h5', 
    monitor='val_loss', 
    verbose=1, 
    save_best_only=True, 
    mode='min' # We want to MINIMIZE loss
)

history = model.fit(
    X_train, y_train,
    epochs=100,  # Let it train for up to 100, EarlyStopping will stop it
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint, early_stopper]
)

# 8. Evaluate the final model
print("Loading best model and evaluating...")
model.load_weights('isl_cnn_model.h5')
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"\nTraining Complete!")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
print(f"Final, most powerful model saved as 'isl_cnn_model.h5'")