import os
import pandas as pd
import numpy as np

from keras.models import load_weights

# Model Directory
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CURR_DIR, "model")
PRO_DIR = os.path.join(CURR_DIR, "processed_data")

# Load model
model = load_weights(MODEL_DIR)

# Load test data
test = pd.read_csv(os.path.join(PRO_DIR, "test.csv"))

test_data = test.copy()

# predict
predictions = model.predict(test_data.values)

prediction_arr = np.array(predictions)

target = [str(int(num)) + '_y' + str(int(i)) for num in test.id for i in range(1, 34)]

submission_df = pd.DataFrame(target, columns=['id'])

prediction_list = [pred for sublist in prediction_arr.tolist() for pred in sublist]

pred_df = pd.DataFrame(prediction_list, columns=['pred'])

test_prediction_df = pd.concat([submission_df, pred_df], axis=1)

os.makedirs(os.path.join(CURR_DIR, "results"), exist_ok=True)
test_prediction_df.to_csv(os.path.join(CURR_DIR, 'results', 'test_prediction.csv'),index=None)