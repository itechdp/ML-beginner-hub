import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import confusion_matrix , accuracy_score, classification_report

rawfile = r"S:\\Programming languages\\Machine Learning\\02_penguine_dataset\\penguins_lter.csv"
dataset = pd.read_csv(rawfile)

columns = list(dataset.columns)
columns = ['studyName', 'Sample Number', 'Region', 'Island', 'Stage', 'Clutch Completion', 'Date Egg', 'Culmen Length (mm)',
           'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)', 'Delta 15 N (o/oo)', 'Delta 13 C (o/oo)', 'Sex', 'Species']

dataset = dataset[columns]
print(dataset)
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 6:12])
x[:, 6:12] = imputer.transform(x[:, 6:12])

label_enc = LabelEncoder()
for iter_col in range(1, 6):
    x[:, iter_col] = label_enc.fit_transform(x[:, iter_col])
x[:, 12] = label_enc.fit_transform(x[:, 12])
y = label_enc.fit_transform(y)

scaler = StandardScaler()
x[:, 5:12] = scaler.fit_transform(x[:, 5:12])

x = pd.DataFrame(x)
y = pd.DataFrame(y)
print(x, y)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=10, test_size=30)

params = {'C':[2,4,6,8,10],'max_iter':[50,100,150]}

Lg = LogisticRegression(C=50,max_iter=75)
lg_model = GridSearchCV(Lg,params,cv=5)
lg_model.fit(x_train,y_train)
y_predicted = lg_model.predict(x_test)
print(lg_model.best_params_)
print(lg_model.best_score_)

confusion_mat = confusion_matrix(y_test,y_predicted)
print(confusion_mat)

report = classification_report(y_test,y_predicted)
print(report)

score = accuracy_score(y_test,y_predicted)
print(score)
