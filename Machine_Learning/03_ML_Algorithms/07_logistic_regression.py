from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix , classification_report , accuracy_score
import pandas as pd


df = load_breast_cancer()

# X is storing all the data and all independent data as well as feature names which is column
x = pd.DataFrame(df['data'],columns=df['feature_names'])

# Y is taking all the data of last column which is dependent data
y = pd.DataFrame(df['target'],columns=['Result'])

#Checking whether the data is balneced or imbalenced if the dataset is imbalenced we need to do upsampeling.
print(y['Result'].value_counts())

#Applying train test split on the dataset varibles which is x and y
x_train , x_test , y_train , y_test = train_test_split(x,y, random_state=50,test_size=0.30)

#This parameter c is regulization to which is going to take by the gridsearch cv and then it will perform f1 or f2 score based on the c values and given prams
params = {'C':[2,8,12],'max_iter':[150,250]}
Lg = LogisticRegression(C=50,max_iter=200)

#Giving model , paramas for formula , scoring we are taking as f1 where we are focusing to decrease the value of false negative and false postitive ,cv = 5
Lg_model = GridSearchCV(Lg,params,scoring='f1',cv=5)

#Fitting my data into model
Lg_model.fit(x_train,y_train)

#Getting best pramaneter which is taken for the grid search cv and the getting best score based on the taken pramas.
print(Lg_model.best_params_)
print(Lg_model.best_score_)

#Predicting the values with respect to x test
y_predicted = Lg_model.predict(x_test)

# Taking the confusion matrix with respect to y predicted values.
confusion_mat = confusion_matrix(y_test,y_predicted)
print(confusion_mat)

#Giving accuracy model based on y test and y predicted values
accuracy_Lg_model = accuracy_score(y_test,y_predicted)
print(accuracy_Lg_model)

#Getting the classification overall report based on the y test values and y predicted values.
Lg_model_report = classification_report(y_test,y_predicted)
print(Lg_model_report)

