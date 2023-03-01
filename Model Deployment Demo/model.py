import pandas as pd

# import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score,recall_score,roc_auc_score
import numpy as np


df = pd.read_csv("Balanced_credit_Risk.csv")

numerical_features = ['person_age', 'person_income',
                      'person_emp_length','loan_amnt', 'loan_int_rate',
       'loan_percent_income',
       'cb_person_cred_hist_length']

X = df[numerical_features]
Y=df[['loan_status']]

from sklearn.model_selection import train_test_split

# This is explicitly used for undersampling.
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

params = {
         'objective': 'binary:logistic',
         'eval_metric': 'error',
         'alpha': 5,
         'nthread': 5,
         'verbosity': 1}

# model = xgb.XGBClassifier(**params,
#                             max_depth = 10,
#                             learning_rate = 0.09,
#                             use_label_encoder=False, n_estimators=200).fit(X,Y)


model = LogisticRegression().fit(X,Y)

# Save your model
import joblib
joblib.dump(model, 'model.pkl')
print("Model dumped!")

# Load the model that you just saved
model = joblib.load('model.pkl')

# Saving the data columns from training
model_columns = list(X.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")

