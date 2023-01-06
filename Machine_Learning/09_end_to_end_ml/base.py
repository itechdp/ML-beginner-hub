import os
import tarfile
import urllib.request

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

fetch_housing_data()

import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()
head = housing.head()
info = housing.info()
ocean_cat_count = housing["ocean_proximity"].value_counts()
describe_housing = housing.describe()

import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))


import numpy as np
# For illustration only. Sklearn has train_test_split()
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
train_set, test_set = split_train_test(housing, 0.2)
len(train_set)
len(test_set)

from zlib import crc32
def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

# Spilliting the data into train and test dataset
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print(test_set.head())

# Applying the stratified shuffled sampeling
housing["median_income"].hist()
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
housing["income_cat"].value_counts()
housing["income_cat"].hist()

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
print(housing["income_cat"].value_counts() / len(housing))

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# Discover and Visualize the Data to Gain Insights
housing = strat_train_set.copy()

# Visualizing Geographical Data
housing.plot(kind="scatter", x="longitude", y="latitude")
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population", figsize=(10,7),
             c="median_house_value", cmap="jet", colorbar=True,
             sharex=False)
plt.legend()

# Looking for Correlations
corr_matrix = housing.corr()
cor_mat_house_value = corr_matrix["median_house_value"].sort_values(ascending=False)

from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))    
housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1)
plt.axis([0, 16, 0, 550000])

# Experimenting with Attribute Combinations
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]
corr_matrix = housing.corr()
corr_matrix_ind =corr_matrix["median_house_value"].sort_values(ascending=False)

# Prepare the Data for Machine Learning Algorithms
housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
housing_labels = strat_train_set["median_house_value"].copy()
# Data Cleaning
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)

# alternatively: housing_num = housing.select_dtypes(include=[np.number])
imputer.fit(housing_num)
housing_med = housing_num.median().values

X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing.index)

# Handling Text and Categorical Attributes
housing_cat = housing[["ocean_proximity"]]

from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
# housing_cat_1hot.toarray() print to numpy array

cat_encoder = OneHotEncoder(sparse=False)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

from sklearn.base import BaseEstimator, TransformerMixin

# column index
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

# Transformation Pipelines
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)

from sklearn.compose import ColumnTransformer
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)

# Applying the model on our dataset
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)

# Let us try to implement all the things with some data from our dataset
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Prediction:",lin_reg.predict(some_data_prepared))
print("Labels:",list(some_labels))

# Applying the rsme on the training set to get the error
from sklearn.metrics import mean_squared_error
housing_prediction = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels,housing_prediction)
lin_rsme = np.sqrt(lin_mse)
print("Linear RSME:",lin_rsme)

# After the calculating the rsme we have found our model is giving us underfitting model, which is
# When  this  happens  it  can  mean  that  the  features  do  not  provide
# enough  information  to  make  good  predictions,  or  that  the  model  is  not  powerful enough 

# This model is not regulized model means it is not containing the method of loss fuction so instead of applying the method
# we will be trying to give more features as well as we could change the algorithm and choose powerful model

# whole dataset rsme < validation datasetrmse = overfitting
# std  high = good prediction
# std low = average/bad prediction

# negative mean squared error = RMSE
# RMSE of validation less = Model working good if higher than it is not working good on validation set or training set.


from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared,housing_labels)

# Finding the value of rsme
housing_prediction = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels,housing_prediction)
tree_resme = np.sqrt(tree_mse)
print("Decision Tree RSME:",tree_resme)

# as we found decision tree rsme as 0.0 now this could be overfitting so we are trying to apply cross validation to
# check whether my data is overfitted or not.

from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg,housing_prepared,housing_labels,scoring="neg_mean_squared_error",cv=10)
tree_rsme_scores = np.sqrt(-scores)


#  high standard deviation means that the scores are spread out over a large range and are therefore less consistent, 
# while a low standard deviation means that the scores are more similar to each other and are more consistent.
# Looking at the result
def display_score(scores):
    print("Scores:",scores)
    print("Mean",scores.mean())
    print("Standard Deviation:",scores.std())


# Checking the score of deicision tree model
display_score(tree_rsme_scores)

lin_scores = cross_val_score(lin_reg,housing_prepared,housing_labels,scoring="neg_mean_squared_error",cv=10)
lin_rsme_scores = np.sqrt(-scores)
display_score(lin_rsme_scores)

# Trying to check the model socre or the accuracy on random forest model
# Gridsearchcv is used to get the best combination of hyper parameter tunning to fine tune (to make our model much accurate)
# GSCV is helping us to find the best combination of Hyper pararmeter automatically 

# Score we are getting when we apply random forest regressor as algorithm on our data
# >>> display_scores(forest_rmse_scores)
# Scores: [49519.80364233 47461.9115823  50029.02762854 52325.28068953
#  49308.39426421 53446.37892622 48634.8036574  47585.73832311
#  53490.10699751 50021.5852922 ]
# Mean: 50182.303100336096
# Standard deviation: 2097.0810550985693



from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forst_reg = RandomForestRegressor()
grid_search = GridSearchCV(forst_reg,param_grid=param_grid,cv=5,scoring="neg_mean_squared_error",return_train_score=True)
grid_search.fit(housing_prepared,housing_labels)
print(grid_search.best_params_) # Getting best params after applying cross validation on all the folds
print(grid_search.best_estimator_) # Getting the best esitmator for our model

# Evaluation score: checking the accuracy score on each round of cross validation 
cvres = grid_search.cv_results_
for mean_scores,params in zip(cvres['mean_test_score'],cvres['params']):
    print(np.sqrt(-mean_scores),params)

# In machine learning, a hyperparameter is a parameter that is set before training a model. 
# Hyperparameters are different from the parameters of a model, which are learned from data during training. 
# The search space of hyperparameters refers to the set of possible values that a hyperparameter can take.

# Randomized Search has two main benefits:

# It is faster than Grid Search, especially when the hyperparameter search space is large. 
# This is because it only evaluates a given number of random combinations, 
# rather than trying out all possible combinations like Grid Search does.

# It has the potential to find better hyperparameter combinations,
# because it is not limited to searching through a predefined set of combinations. 
# Grid Search can only find the best combination within the set of combinations it evaluates,
# whereas Randomized Search has the ability to potentially find a better combination
#  that is outside of the set of combinations it evaluates.

# Analyze the Best Models and Their Errors

# Getting the score of each model's each attribute accuracy to check the importance of the feature
feature_importance = grid_search.best_estimator_.feature_importances_
print(feature_importance)

# importance scores next to their corresponding attribute name
# extra_attrib = ['rooms_per_hhold','pop_per_hhold','bedrooms_per_room']
# cat_encoder = full_pipeline.named_transformers_('cat')
# cat_one_hot_attribs = list(cat_encoder.categories_[0])
# attributes = num_attribs + extra_attrib + cat_one_hot_attribs
# sorted_zip = sorted(zip(feature_importance,attributes),reverse=True)
# print(sorted_zip)

# Evaluating the final model
final_model = grid_search.best_estimator_

x_test = strat_test_set.drop("median_house_value",axis=1)
y_test = strat_test_set['median_house_value'].copy()

x_test_prepared = full_pipeline.transform(x_test)
final_prediction = final_model.predict(x_test_prepared)

final_mse = mean_squared_error(y_test,final_prediction)
final_rmse = np.sqrt(final_mse)
print((final_rmse))

# The confidence of lauching the model by looking at the final rmse is not too much satisfied for that you need to 
# compute to check whether this model is ready to lauch or not

from scipy import stats
confidence = 0.95
squared_error = (final_prediction-y_test) ** 2
final_stats = np.sqrt(stats.t.interval(confidence,len(squared_error)-1,loc=squared_error.mean(),scale=stats.sem(squared_error)))
print(final_stats)