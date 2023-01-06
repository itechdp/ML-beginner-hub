import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

house_X_train = [[500],[700],[900],[1100],[1300],[1500],[1700],[1900],[2100]]
house_X_test = [[600],[800],[1000],[1200],[1400],[1600],[1800],[2000],[2200]]

house_Y_train = [[12],[14],[19],[25],[27],[30],[40],[50],[60]]
house_Y_test = [[13],[18],[21],[26],[29],[35],[45],[55],[65]]

model = linear_model.LinearRegression()
model.fit(house_X_train,house_Y_train)
house_Y_predicted = model.predict(house_X_test)

print("Mean squared error:",mean_squared_error(house_Y_test,house_Y_predicted))
print("Weight:",model.coef_)
print("Intercept: ",model.intercept_)

plt.scatter(house_X_test,house_Y_test)
plt.plot(house_X_test,house_Y_predicted)
plt.show()