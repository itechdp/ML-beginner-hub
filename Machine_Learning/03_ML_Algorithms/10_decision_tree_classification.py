from sklearn import tree
from sklearn.datasets import load_iris

df = load_iris()
x ,y = df.data , df.target
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x,y)
tree.plot_tree(clf)