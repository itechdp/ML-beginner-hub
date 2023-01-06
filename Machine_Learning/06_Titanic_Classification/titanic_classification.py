from itertools import count
import pandas as pd 
import numpy as np 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler , LabelEncoder 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt 

raw_file_train = r"E:\\Programming languages\\Python\\Machine Learning\\06_Titanic_Classification\\train.csv"
raw_file_test = r"E:\\Programming languages\\Python\\Machine Learning\\06_Titanic_Classification\\test.csv"
dataset_train = pd.read_csv(raw_file_train)
dataset_test = pd.read_csv(raw_file_test)


#Pre Processing for training dataset 
for columns in dataset_train.columns: 
    dataset_train[columns] = dataset_train[columns].replace("\W"," ",regex=True)

dataset_train_columns = list(dataset_train.columns)
dataset_train_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked','Survived']
dataset_train = dataset_train[dataset_train_columns]
x_train = dataset_train.iloc[:,:-1].values
y_train = dataset_train.iloc[:,-1].values
print(dataset_train.isnull().sum())

label_enc = LabelEncoder()
x_train[:,6]= label_enc.fit_transform(x_train[:,6])
x_train[:,7]= label_enc.fit_transform(x_train[:,7])
x_train[:,1]= label_enc.fit_transform(x_train[:,1])


imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
imputer.fit(x_train[:,2:3])
imputer.fit(x_train[:,6:7])
imputer.fit(x_train[:,7:8])
x_train[:,6:7] = imputer.transform(x_train[:,6:7])
x_train[:,2:3] = imputer.transform(x_train[:,2:3])
x_train[:,7:8] = imputer.transform(x_train[:,7:8])

scaler = StandardScaler()
x_train[:,2:3] = scaler.fit_transform(x_train[:,2:3])
x_train[:,5:8] = scaler.fit_transform(x_train[:,5:8])

#pre processing for training dataset
for columns in dataset_test.columns: 
    dataset_test[columns] = dataset_test[columns].replace("\W"," ",regex=True)

dataset_test_columns = list(dataset_test.columns)
dataset_test_columns = ['Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']
dataset_test = dataset_test[dataset_test_columns]
x_test = dataset_test.iloc[:,:].values

label_enc = LabelEncoder()
x_test[:,6]= label_enc.fit_transform(x_test[:,6])
x_test[:,1]= label_enc.fit_transform(x_test[:,1])
x_test[:,7] = label_enc.fit_transform(x_test[:,7])


imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
imputer.fit(x_test[:,2:3])
imputer.fit(x_test[:,6:7])
x_test[:,6:7] = imputer.transform(x_test[:,6:7])
x_test[:,2:3] = imputer.transform(x_test[:,2:3])


scaler = StandardScaler()
x_test[:,2:3] = scaler.fit_transform(x_test[:,2:3])
x_test[:,5:7] = scaler.fit_transform(x_test[:,5:7])

x_train = pd.DataFrame(x_train)
y_train = pd.DataFrame(y_train)
x_test =  pd.DataFrame(x_test)
x_test.dropna(inplace=True)

Knn_class = KNeighborsClassifier(n_neighbors=5)
params = {'n_neighbors':[5,10,15,20,25,50,100,150,200]}
knn_cv_mod = GridSearchCV(Knn_class,params,cv=10)
knn_cv_mod.fit(x_train,y_train)
y_predicted = knn_cv_mod.predict(x_test)
x_test["Survived"] = y_predicted
print(x_test)

print(y_predicted)

y_predicted = list(y_predicted)
counttotal = []
counttotal.insert(0,y_predicted.count(0))
counttotal.insert(1,y_predicted.count(1))
labels = ["Not survived",'Survived']
explode = [0,0.05]
color = ['red','green']
plt.pie(counttotal,labels=labels,shadow=True,explode=explode,colors=color,autopct='%1.1f%%',wedgeprops={'edgecolor':'black'})
plt.title("Titanic People")
plt.tight_layout()
plt.show()