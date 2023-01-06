import matplotlib.pyplot as plt
import numpy as np 
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()
#keys getting all the features of the data sets.
print(diabetes.DESCR)

#This below line is choosing one fearture from the datasets.
diabetes_X = diabetes.data[:,np.newaxis,2]

#Below two lines are slicing the data for training and testing of independent varialbe.
diabetes_X_train = diabetes_X[:-10]
diabetes_X_test = diabetes_X[-10:]
# print(diabetes_X_train,diabetes_X_test)
print(x_train.shape , x_test.shape)

#Below two lines are slicing the data for training and testing of Dependent varialbe.
diabetes_Y_train = diabetes.target[:-10]
diabetes_Y_test = diabetes.target[-10:]
# We are using linear regression algorithm for this dataset where we are assinging the linear graph to variable model
# model.fit(taking X axis data as argument to train the data , taking Y axis data as argument to train the data)
# this model.fit function is plotting the training data on the graph and passing the best fit line
# diabetes_Y_predicted is storing the predicted values on the graph 

model = linear_model.LinearRegression()
model.fit(diabetes_X_train,diabetes_Y_train)
diabetes_Y_predicted = model.predict(diabetes_X_test)
# print(diabetes_Y_predicted)

# mean_squared_error() is subtraing the predicted values and actual values and finding the average of all the predicted values.
# model.coef_ is finding the weight or slope of the model
# model.intercept_ is finding the intercept of the model
print("Mean squared error is: ", mean_squared_error(diabetes_Y_test, diabetes_Y_predicted))
print("Weights: ",model.coef_)
print("Intercept: ",model.intercept_)

# plt.scatter is plotting the values of testing data.
# plt.plot is plotting the best fit line 
# plt.show() displaying the graph
plt.scatter(diabetes_X_test , diabetes_Y_test)
plt.plot(diabetes_X_test,diabetes_Y_predicted)
plt.show()