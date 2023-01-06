from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples , silhouette_score

X,y = make_blobs(n_samples=500,n_features=5,centers=4,cluster_std=1,center_box=(-10,10),shuffle=True,random_state=1)
wcss = []
print(X.shape)
k_means_mod = KMeans(X,init='k-means++',random_state=1)
k_means_mod.fit(X)
print(k_means_mod.inertia_)

#Hello