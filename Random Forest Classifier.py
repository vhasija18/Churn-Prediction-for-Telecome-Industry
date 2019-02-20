# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
data = pd.read_csv(r'C:\Users\hasij\Downloads\Data.csv')
print(data.head(5))
print(data.info())

gender_d = {'Male':1, 'Female':2}
data['gender'] = data['gender'].apply(lambda x:gender_d[x])

y_n_d = {'Yes':1, 'No': 0}
data['Partner'] = data['Partner'].apply(lambda x: y_n_d[x])
data['Dependents'] = data['Dependents'].apply(lambda x: y_n_d[x])
data['PhoneService'] = data['PhoneService'].apply(lambda x: y_n_d[x])
data['PaperlessBilling'] = data['PaperlessBilling'].apply(lambda x: y_n_d[x])
data['Churn'] = data['Churn'].apply(lambda x: y_n_d[x])

tech_spt_d = {'Yes':1, 'No':0, 'No internet service': 3, 'No phone service':4}
data['TechSupport'] = data['TechSupport'].apply(lambda x: tech_spt_d[x])
data['OnlineSecurity'] = data['OnlineSecurity'].apply(lambda x: tech_spt_d[x])
data['OnlineBackup'] = data['OnlineBackup'].apply(lambda x: tech_spt_d[x])
data['DeviceProtection'] = data['DeviceProtection'].apply(lambda x: tech_spt_d[x])
data['StreamingTV'] = data['StreamingTV'].apply(lambda x: tech_spt_d[x])
data['StreamingMovies'] = data['StreamingMovies'].apply(lambda x: tech_spt_d[x])
data['MultipleLines'] = data['MultipleLines'].apply(lambda x: tech_spt_d[x])

Contract_d = {'Month-to-month':1, 'One year':2, 'Two year':3}
data['Contract'] = data['Contract'].apply(lambda x: Contract_d[x])

Internet_d = {'DSL':1, 'Fiber optic':2, 'No':3}
data['InternetService'] = data['InternetService'].apply(lambda x: Internet_d[x])

Payment_d = {'Electronic check':1, 'Mailed check':2, 'Bank transfer (automatic)': 3, 'Credit card (automatic)':4}
data['PaymentMethod'] = data['PaymentMethod'].apply(lambda x: Payment_d[x])

data['TotalCharges'] = pd.to_numeric(data['TotalCharges'],errors = 'coerce').fillna(0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
data[['TotalCharges','MonthlyCharges','tenure']] = sc.fit_transform(data[['TotalCharges','MonthlyCharges','tenure']])

X = data.iloc[:,1:19].values
y = data.iloc[:,20].values
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=90, max_depth=7,random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score
print ("accuracy_score")
print (accuracy_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix
print ("confusion matrix")
print (confusion_matrix(y_test, y_pred))

from sklearn.metrics import f1_score
print ("f1_score")
print (f1_score(y_test, y_pred))

from sklearn.metrics import roc_auc_score
print ("roc_auc_score")
print (roc_auc_score(y_test, y_pred))
