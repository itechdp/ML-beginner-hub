#Linear regression House pricing dataset
from sklearn.datasets import load_boston
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score , train_test_split
from sklearn.metrics import r2_score


#Loading the dataset
df = load_boston()

#converting the dataset into the table form
dataset = pd.DataFrame(df.data)
print(dataset.head())

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
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.23,random_state=30)

#applying the cross validation on dependent and independent feature where it will do train and test split and it is diving the 
# independent and dependent feature in parts and whatever is giving best accuracy it will combine
#parameter: model , x inde-feature , y depen-features, scroing = mse or neg mse , cross validation iteration
mse = cross_val_score(lin_reg,x_train,y_train,scoring='neg_mean_squared_error',cv=5)

#average of mse
mean_mse = np.mean(mse)
print(mean_mse)

#Fitting the training data with respect to x and y
lin_reg.fit(x_train,y_train)

# Predicting the value so we can get to know that how good is our model 
y_pred = lin_reg.predict(x_test)

# Giving y test and y pred to get the percentage of model accuracy
r2_score1 = r2_score(y_pred,y_test)
print(r2_score1)
