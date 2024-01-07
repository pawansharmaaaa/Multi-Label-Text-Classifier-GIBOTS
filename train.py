import os
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Set working directory
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
PRO_DIR = os.path.join(CURR_DIR, "processed_data")

# MODEL HYPERPARAMETERS:
no_of_epochs = 100
batch_size = 100

# Read processed data:

train = pd.read_csv(os.path.join(PRO_DIR, "train.csv"))
test = pd.read_csv(os.path.join(PRO_DIR, "test.csv"))
labels = pd.read_csv(os.path.join(PRO_DIR, "trainLabels.csv"))

# Starting the model building process
model = Sequential()
model.add(Dense(5000, activation='relu', input_shape=(train.shape[1],)))
model.add(Dense(2500, activation='relu'))
model.add(Dense(1250, activation='relu'))
model.add(Dense(625, activation='relu'))
model.add(Dense(33, activation='sigmoid')) # Adapt output layer size for 33 labels

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train.values, labels.values, epochs=10, batch_size=100, verbose=1)

# Save model
os.makedirs(os.path.join(CURR_DIR, "models"), exist_ok=True)
model.save(os.path.join(CURR_DIR, 'models', "model.h5"))