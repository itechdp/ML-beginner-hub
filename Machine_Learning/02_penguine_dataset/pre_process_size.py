import pandas as pd 
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder , LabelEncoder
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
#Data loading 
rawfile = r"S:\\Programming languages\\Machine Learning\\02_penguine_dataset\\penguins_size.csv"
dataset = pd.read_csv(rawfile)

# Assigining dependent and independent columns
x = dataset.iloc[:,1:].values
y = dataset.iloc[:,0].values
#Checking for the null values
null_val = dataset.isnull().sum()

#Data encoding
column_enc = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
x = np.array(column_enc.fit_transform(x))

#Label binary encode for x and y
label_enc = LabelEncoder()
x[:,-1] = label_enc.fit_transform(x[:,-1])
y = label_enc.fit_transform(y)

# Finding NUll values 
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(x[:,3:7])
x[:,3:7] = imputer.transform(x[:,3:7])

#Coverting array to dataframe
x = pd.DataFrame(x)
y = pd.DataFrame(y)
# Model formation

x_train, x_test, y_train ,y_test = train_test_split(x,y,random_state=23,test_size=0.30)
params = {'n_neighbors':[2,3,5,10,15],'p':[2]}
knn_mod = KNeighborsClassifier(n_neighbors=5,p=2,weights='distance')
knn_cv = GridSearchCV(knn_mod,params,cv=5)
knn_cv.fit(x_train,y_train)
y_predicted = knn_cv.predict(x_test)
plt.hist(y_predicted)
plt.show()