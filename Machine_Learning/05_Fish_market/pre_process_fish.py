import numpy as np 
import pandas as pd 
from sklearn.preprocessing import  LabelEncoder,StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
rawfile = r"E:\\Programming languages\\Python\\Machine Learning\\05_Fish_market\\Fish.csv"
dataset = pd.read_csv(rawfile)

columns = list(dataset.columns)
columns = ['Species','Width','Length1','Length2', 'Length3', 'Height','Weight']
dataset = dataset[columns]

x = dataset.iloc[:,0:-1].values
y = dataset.iloc[:,-1].values

label_enc = LabelEncoder()
x[:,0] = label_enc.fit_transform(x[:,0])

scaler = StandardScaler()
x = scaler.fit_transform(x)

x = pd.DataFrame(x)
y = pd.DataFrame(y)

lin_reg = LinearRegression()
x_train , x_test , y_train ,y_test = train_test_split(x,y,test_size=0.30,random_state=10)
mse = cross_val_score(lin_reg,x_train,y_train,scoring='neg_mean_squared_error',cv=5)
mse = np.mean(mse)
print(f"Mean squared error: {mse}")

lin_reg.fit(x_train,y_train)
y_predicted = lin_reg.predict(x_test)
r2_score1 = r2_score(y_predicted,y_test)
print(f"R2 Score: {r2_score1}")

