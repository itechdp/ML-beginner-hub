#Linear regression House pricing dataset
from sklearn.datasets import load_boston
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression , Ridge
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import r2_score


#Loading the dataset
df = load_boston()

#converting the dataset into the table form
dataset = pd.DataFrame(df.data)

#We are getting the feature names.
dataset.columns = df.feature_names

#adding one dependent feature with the help of object.name of the column --> dp.target 
dataset['Price'] = df.target

#Dividing the dataset into independet and dependent features.
#iloc[data , starting point : ending point of columns]
x = dataset.iloc[:,:-1]  #indpendent features.
y = dataset.iloc[:,-1] #dependent feautes.

# print(x.head())
# print(y.head())

#Linear regression
lin_reg = LinearRegression()

#applying the cross validation on dependent and independent feature where it will do train and test split and it is diving the 
# independent and dependent feature in parts and whatever is giving best accuracy it will combine
#parameter: model , x inde-feature , y depen-features, scroing = mse or neg mse , cross validation iteration
mse = cross_val_score(lin_reg,x,y,scoring='neg_mean_squared_error',cv=5)

#average of mse
mean_mse = np.mean(mse)
print(mean_mse)

# lin_reg.predict(give test data) and it will predict

Ridge_model = Ridge()
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=30,test_size=0.30)
#Applying the hyperparameter alpha as parameter to change the feature or steepness of slope
Params = {'alpha':[1e-15,1e-10,1e-8,1e-5,1e-3,1e-2,0,1,3,5,8,10]}

#Grid search cv where we are adding our model and hyperparameter and then we are trying to find the MSE with cross validation 
ridge_reg = GridSearchCV(Ridge_model,Params,scoring='neg_mean_squared_error',cv=5)
ridge_reg.fit(x_train,y_train)

# Below two lines is showing us the hyperparameter picked value and best MSE which we have got.
print(ridge_reg.best_params_)
print(ridge_reg.best_score_)

y_pred = ridge_reg.predict(x_test)
r2_score1 = r2_score(y_pred,y_test)
print(r2_score1)



