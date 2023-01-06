Knn_class = KNeighborsClassifier(n_neighbors=5)
params = {'n_neighbors':[5,10,15,20,25,50,100,150,200]}
knn_cv_mod = GridSearchCV(Knn_class,params,cv=10)
knn_cv_mod.fit(x_train,y_train)
y_predicted = knn_cv_mod.predict(x_test)
print(y_predicted)