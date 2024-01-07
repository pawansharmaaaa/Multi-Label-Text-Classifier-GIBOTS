import os
import pandas as pd

from fancyimpute import IterativeImputer

# Set working directory
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
UNP_DIR = os.path.join(CURR_DIR, "unprocessed_data")
PRO_DIR = os.path.join(CURR_DIR, "processed_data")

def check_null_values(df):
    return df.isna().sum().sort_values(ascending=False)*100/len(df)

# Read unprocessed data
train = pd.read_csv(os.path.join(UNP_DIR, "train.csv"))
test = pd.read_csv(os.path.join(UNP_DIR, "test.csv"))
labels = pd.read_csv(os.path.join(UNP_DIR, "trainLabels.csv"))

# Adding colum names to test data
test.columns = train.columns

# Checking null values
train_null = check_null_values(train)
train_labels_null = check_null_values(labels)
test_null = check_null_values(test)

# Checking data types and creating lists of numeric and object columns
numeric_cols = train.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns.to_list()
object_cols = train.select_dtypes(include=object).columns

train[object_cols] = train[object_cols].astype(str)
test[object_cols] = test[object_cols].astype(str)

boolean_cols = [col for col in object_cols if train[col][0].isupper()]
alphanumeric_cols = [col for col in object_cols if not train[col][0].isupper()]

train[boolean_cols] = train[boolean_cols].replace({'YES': 1, 'NO': 0, 'nan': None})
test[boolean_cols] = test[boolean_cols].replace({'YES': 1, 'NO': 0, 'nan': None})

# IMPUTING MISSING VALUES

# Initialize MICE imputer
num_imp = IterativeImputer(initial_strategy='mean',max_iter=10)
bool_imp = IterativeImputer(initial_strategy='most_frequent',max_iter=10)

# Fit and transform the train data
train[numeric_cols] = num_imp.fit_transform(train[numeric_cols])
train[boolean_cols] = bool_imp.fit_transform(train[boolean_cols])
train = train[numeric_cols+boolean_cols]

# Transform the test data
test[numeric_cols] = num_imp.transform(test[numeric_cols])
test[boolean_cols] = bool_imp.transform(test[boolean_cols])

# Save processed data
train[numeric_cols+boolean_cols].to_csv(os.path.join(PRO_DIR, "train.csv"), index=False)
test[numeric_cols+boolean_cols].to_csv(os.path.join(PRO_DIR, "test.csv"), index=False)
labels.loc[:9998, 'y1':].to_csv(os.path.join(PRO_DIR, "trainLabels.csv"), index=False)