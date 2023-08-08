import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import openpyxl
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
warnings.simplefilter(action='ignore', category=Warning)

df=pd.read_excel("data/panic_data.xlsx")

df.head()


df.info()

df.dropna()

data=df.drop(["secureid","timestamp"], axis=1)

data.head()


columns=data.columns

columns

for i in columns:
  plt.figure(figsize=(15,6))
  sns.countplot(x=data[i], data=data)
  plt.show()


data_kategorik=data.select_dtypes(include=["object"])

data_columns=data_kategorik.columns
data_columns

from sklearn.preprocessing import LabelEncoder
for i in data_columns:
  lbe = LabelEncoder()
  data[i]=lbe.fit_transform(data[i])


data.head()


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


y=data["Outcome"]
X=data.drop(["Outcome"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)



#Logistic Regression Algorithm

log_model=LogisticRegression().fit(X_train,y_train)
y_pred=log_model.predict(X_test)
log_skor= accuracy_score(y_test,y_pred)
print('Logistic Model Dogrulugu:', log_skor)
cf_matrix_log_model=confusion_matrix(y_test,y_pred)
sns.heatmap(cf_matrix_log_model,annot=True,cbar=False,fmt="g")
log_model_score=accuracy_score(y_test, y_pred)
plt.xlabel("y_pred")
plt.ylabel("y_test")
plt.title("Model Accuracy:"+str(log_model_score))
plt.show()

print("Logistic Regression Sınıflandırma Raporu")
print(classification_report(y_test, y_pred))

#SVM Classification Algorithm

SVM_model=SVC().fit(X_train,y_train)
y_pred=SVM_model.predict(X_test)
SVM_skor= accuracy_score(y_test,y_pred)
print('SVM Model Dogrulugu:', SVM_skor)
cf_matrix_SVM_model=confusion_matrix(y_test,y_pred)
sns.heatmap(cf_matrix_SVM_model,annot=True,cbar=False,fmt="g")
SVM_model_score=accuracy_score(y_test, y_pred)
plt.xlabel("y_pred")
plt.ylabel("y_test")
plt.title("Model Accuracy:"+str(SVM_model_score))
plt.show()
print("SVM Sınıflandırma Raporu")
print(classification_report(y_test,y_pred))

#Decision Tree Algorithms

DT_model=DecisionTreeClassifier().fit(X_train,y_train)
y_pred=DT_model.predict(X_test)
DT_skor= accuracy_score(y_test,y_pred)
print('DT Model Dogrulugu:', DT_skor)
cf_matrix_DT_model=confusion_matrix(y_test,y_pred)
sns.heatmap(cf_matrix_DT_model,annot=True,cbar=False,fmt="g")
DT_model_score=accuracy_score(y_test, y_pred)
plt.xlabel("y_pred")
plt.ylabel("y_test")
plt.title("Model Accuracy:"+str(DT_model_score))
plt.show()

print("DT Sınıflandırma Raporu")
print(classification_report(y_test,y_pred))

#Random Forest

RF_model=RandomForestClassifier().fit(X_train,y_train)
y_pred=RF_model.predict(X_test)
RF_skor= accuracy_score(y_test,y_pred)
print('RF Model Dogrulugu:', RF_skor)
cf_matrix_RF_model=confusion_matrix(y_test,y_pred)
sns.heatmap(cf_matrix_RF_model,annot=True,cbar=False,fmt="g")
RF_model_score=accuracy_score(y_test, y_pred)
plt.xlabel("y_pred")
plt.ylabel("y_test")
plt.title("Model Accuracy:"+str(RF_model_score))
plt.show()

print("Random Forest Sınıflandırma Raporu")
print(classification_report(y_test,y_pred))


#XGBOOST Classification Algorithms

XGBM_model=XGBClassifier().fit(X_train,y_train)
y_pred=XGBM_model.predict(X_test)
XGBM_skor= accuracy_score(y_test,y_pred)
print('XGBM Model Dogrulugu:', XGBM_skor)
cf_matrix_XGBM_model=confusion_matrix(y_test,y_pred)
sns.heatmap(cf_matrix_XGBM_model,annot=True,cbar=False,fmt="g")
XGBM_model_score=accuracy_score(y_test, y_pred)
plt.xlabel("y_pred")
plt.ylabel("y_test")
plt.title("Model Accuracy:"+str(XGBM_model_score))
plt.show()
print("XGBOOST Sınıflandırma Raporu")
print(classification_report(y_test,y_pred))


LGBM_model=LGBMClassifier().fit(X_train,y_train)
y_pred=LGBM_model.predict(X_test)
LGBM_skor= accuracy_score(y_test,y_pred)
print('RF Model Dogrulugu:', LGBM_skor)
cf_matrix_LGBM_model=confusion_matrix(y_test,y_pred)
sns.heatmap(cf_matrix_LGBM_model,annot=True,cbar=False,fmt="g")
LGBM_model_score=accuracy_score(y_test, y_pred)
plt.xlabel("y_pred")
plt.ylabel("y_test")
plt.title("Model Accuracy:"+str(LGBM_model_score))
plt.show()

print("LGBM Sınıflandırma Raporu")
print(classification_report(y_test,y_pred))

import pickle
pickle.dump(LGBM_model,open('model/LGBMmodel.pkl','wb'))


model=pickle.load(open("model/LGBMmodel.pkl", "rb"))

test_data=pd.read_excel("data/test_panic.xlsx")

test_data=test_data.drop(["secureid","timestamp"], axis=1)

testdata_kategorik=test_data.select_dtypes(include=["object"])

testdata_columns=testdata_kategorik.columns
testdata_columns

from sklearn.preprocessing import LabelEncoder
for i in testdata_columns:
  lbe = LabelEncoder()
  test_data[i]=lbe.fit_transform(test_data[i])
test_data.head()

testy=test_data["Outcome"]
testX=test_data.drop(["Outcome"], axis=1)

testy_pred=model.predict(testX)
skor= accuracy_score(testy,testy_pred)
print(' Model Dogrulugu:', skor)
cf_matrix_model=confusion_matrix(testy,testy_pred)
sns.heatmap(cf_matrix_model,annot=True,cbar=False,fmt="g")
XGBM_model_score=accuracy_score(testy,testy_pred)
plt.xlabel("y_pred")
plt.ylabel("y_test")
plt.title("Model Accuracy:"+str(skor))
plt.show()
print("Sınıflandırma Raporu")
print(classification_report(testy,testy_pred))