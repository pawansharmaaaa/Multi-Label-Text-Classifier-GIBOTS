import os
import pandas as pd
import numpy as np

# import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import F1Score
from sklearn.preprocessing import StandardScaler

# Set working directory
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
PRO_DIR = os.path.join(CURR_DIR, "processed_data")

# MODEL HYPERPARAMETERS:
no_of_epochs = 100
batch_size = 32

# Read processed data:

train = pd.read_csv(os.path.join(PRO_DIR, "train.csv"))
test = pd.read_csv(os.path.join(PRO_DIR, "test.csv"))
labels = pd.read_csv(os.path.join(PRO_DIR, "trainLabels.csv"))

############################## TRAIN MODEL ##############################

# Feature scaling
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train.values)

# Model architecture with reduced first layer, regularization, and early stopping
model = Sequential()
model.add(Dense(1000, activation='relu', input_shape=(train_scaled.shape[1],)))  # Reduced first layer
model.add(Dropout(0.2))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(33, activation='sigmoid'))  # Output layer for 33 labels

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', F1Score()])  # Include F1-score

# Early stopping
early_stopping = EarlyStopping(patience=10, restore_best_weights=True)

# Training with hyperparameter tuning (adjust epochs and batch size as needed)
model.fit(train_scaled, labels.values.astype(np.float32), epochs=no_of_epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping])

# Save model
os.makedirs(os.path.join(CURR_DIR, "model"), exist_ok=True)
model.save(os.path.join(CURR_DIR, 'model'))