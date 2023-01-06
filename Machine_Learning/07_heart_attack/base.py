import pandas as pd 
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt

rawfile = r"E:\\Programming languages\\Python\\Machine_Learning\\07_heart_attack\\heart.csv"
dataset = pd.read_csv(rawfile)
print(dataset)

x_train = dataset.iloc[:,:-1].values
y_train = dataset.iloc[:,-1].values

x_test = dataset.iloc[:,:-1]

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)

x_train = pd.DataFrame(x_train)
y_train = pd.DataFrame(y_train)

x_test = pd.DataFrame(x_test)

params = {'C':[2,8,12],'max_iter':[150,250]}
lg_model = LogisticRegression(C=50,max_iter=85)
lg_model_grid = GridSearchCV(lg_model,params,cv=5)
lg_model_grid.fit(x_train,y_train) 
y_predicted = lg_model_grid.predict(x_test)

confusion_mat = confusion_matrix(y_train,y_predicted)
accuracy_scr = accuracy_score(y_train,y_predicted)
classification_rpt = classification_report(y_train,y_predicted)

list_pre = []

y_predicted = list(y_predicted)
list_pre.insert(0,y_predicted.count(0))
list_pre.insert(1,y_predicted.count(1))

color = ['green','red']
explode = [0.2,0.0]
label = ["Heart Attack Not-Declared","Heart Attack Declared"]
plt.pie(list_pre,labels=label,shadow=True,autopct="%1.1f%%",explode=explode,colors=color,wedgeprops={'edgecolor':'black'})
plt.title("Heart Attack prediction")
plt.tight_layout()
plt.show()