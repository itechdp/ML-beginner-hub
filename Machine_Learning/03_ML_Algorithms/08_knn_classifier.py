import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.datasets import load_iris
import pandas as pd 


df = load_iris()

x = pd.DataFrame(df['data'],columns=df['feature_names'])
# print(x)
y = pd.DataFrame(df['target'],columns=['Results'])
# print(y['Results'])
# print(y['Results'].value_counts())

x_train , x_test , y_train , y_test = train_test_split(x,y,random_state=1,test_size=0.50)
print(type(x_train ))
params = {'n_neighbors':[2,3,5,10,15],'p':[2]}

Knn_mod = KNeighborsClassifier(n_neighbors=10,p=2,weights='distance')

Knn_mod_cv = GridSearchCV(Knn_mod,params,cv=5)

Knn_mod_cv.fit(x_train,y_train)

# print(Knn_mod_cv.best_params_ , Knn_mod_cv.best_score_)

y_predicted = Knn_mod_cv.predict(x_test)
