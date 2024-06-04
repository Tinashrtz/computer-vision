#train test-split//kfold//leavepout--sesion 7 homework
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


X,y = datasets.load_diabetes(return_X_y = True , as_frame = True)
y = pd.DataFrame(y)
error_list_split=[]
error_list_kfold=[]
error_list_leaveone=[]
X = X.drop(['sex', 's4'], axis=1)
#print(X)
#print(type(X))

#using train test split to evaluate
for i in range(1,50):
    x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=i/50,
                                                        random_state=32)
    model=LinearRegression()
    model.fit(x_train,y_train)
    y_predict=model.predict(x_test)
    error=mean_squared_error(y_test,y_predict)
    error_list_split.append(error)

#using leaveoneout to evalute
selector = LeaveOneOut()
for train_index, test_index in selector.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model=LinearRegression()
    model.fit(X_train,y_train)
    y_predict=model.predict(X_test)
    error=mean_squared_error(y_test,y_predict)
    error_list_leaveone.append(error)

#using kfold to evaluate

selector = KFold(n_splits=4)
for train_index, test_index in selector.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model=LinearRegression()
    model.fit(X_train,y_train)
    y_predict=model.predict(X_test)
    error=mean_squared_error(y_test,y_predict)
    error_list_kfold.append(error)
  
    
    
#print(error_list_split)
#print(error_list_leaveone)
#print(error_list_kfold)
print("train-test-split minimum error=",np.min(error_list_split))
print("leaveoneout minimum error= ",np.min(error_list_leaveone))
print("kfold minimum error= ",np.min(error_list_kfold))
print("so the best way is to train with leave one out because the error is minimum")