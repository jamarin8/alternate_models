import pandas as pd
from smart_open import smart_open

trainpath = 's3://sagemaker-shared-resources/X_train_20210603.csv'
testpath = 's3://sagemaker-shared-resources/X_test_20210603.csv'
X_train = pd.read_csv(smart_open(trainpath), low_memory=False)
X_test = pd.read_csv(smart_open(testpath), low_memory=False)
y_train = X_train['dep_var']
X_train = X_train.drop('dep_var', axis=1)
y_test = X_test['dep_var']
X_test = X_test.drop('dep_var', axis=1)
