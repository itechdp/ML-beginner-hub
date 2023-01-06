from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

#Loading datasets
iris = datasets.load_iris()

#printing description 
print(iris.DESCR)

# Adding features and labels with .data and .target
features = iris.data
labels = iris.target 

#Importing class KNeighborsclassfier()
# clf.fit(adding features , adding labels)
# predicting the object or result as per the given points using clf.predict()
clf = KNeighborsClassifier()
clf.fit(features,labels)
predict = clf.predict([[1,1,1,1]])
print(predict)