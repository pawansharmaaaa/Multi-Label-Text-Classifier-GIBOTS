import os
import pandas as pd
import numpy as np

from keras.models import load_weights

# Model Directory
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CURR_DIR, "models")
PRO_DIR = os.path.join(CURR_DIR, "processed_data")

# Load model
model = load_weights(os.path.join(MODEL_DIR, "model0_859.keras"))

# Load test data
test = pd.read_csv(os.path.join(PRO_DIR, "test.csv"))

test_data = test.copy()

# predict
predictions = model.predict(test_data.values)

prediction_arr=np.array(predictions)

target=[]
for num in test.id:
    for i in range(1,34):
        target.append(str(num)+'_y'+str(i))

submission_df=pd.DataFrame(target,columns=['id'])

prediction_list=[]
for i in range(len(prediction_arr)):
    prediction_list.extend(prediction_arr.tolist()[i])

pred_df=pd.DataFrame(prediction_list,columns=['pred'])

test_prediction_df=pd.concat([submission_df,pred_df],axis=1)

test_prediction_df.to_csv(os.path.join(CURR_DIR, 'results', 'test_prediction.csv'),index=None)