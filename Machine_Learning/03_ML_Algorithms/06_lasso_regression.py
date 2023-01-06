# Linear regression House pricing dataset
from sklearn.datasets import load_boston
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV , train_test_split
from sklearn.metrics import r2_score



# Loading the dataset
df = load_boston()

# converting the dataset into the table form
dataset = pd.DataFrame(df.data)

# We are getting the feature names.
dataset.columns = df.feature_names

# adding one dependent feature with the help of object.name of the column --> dp.target
dataset['Price'] = df.target

# Dividing the dataset into independet and dependent features.
# iloc[data , starting point : ending point of columns]
x = dataset.iloc[:, :-1]  # indpendent features.
y = dataset.iloc[:, -1]  # dependent feautes.

# print(x.head())
# print(y.head())

x_train , x_test , y_train, y_test = train_test_split(x,y,random_state=50,test_size=0.30)

Lasso_model = Lasso()

# Applying the hyperparameter alpha as parameter to change the feature or steepness of slope
Params = {'alpha': [1e-15, 1e-10, 1e-8, 1e-5, 1e-3, 1e-2,
                    0, 1, 3, 5, 8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}

# Grid search cv where we are adding our model and hyperparameter and then we are trying to find the MSE with cross validation
Lasso_reg = GridSearchCV(
    Lasso_model, Params, scoring='neg_mean_squared_error', cv=5)
Lasso_reg.fit(x_train, y_train)

# best pramas is showing us which alpha value has picked up for the computing the mathematical opeation
print(Lasso_reg.best_params_)

# It is showing us the best MSE it should be near to 0
print(Lasso_reg.best_score_)

Lasso_model.fit(x_train,y_train)
y_pred = Lasso_reg.predict(x_test)
r2_score1 = r2_score(y_pred,y_test)
print(r2_score1)