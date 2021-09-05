import numpy as np
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage
from scipy.spatial import KDTree

data = np.random.rand(10, 2)

model = KMeans(n_clusters=2)

model.fit(data)

print("*************Data*************")
print(data)
print("*************Labels*************"):
print(model.labels_)
print("*************Centers*************")
print(model.cluster_centers_)
print("*************Distances*************")
print(model.transform(data))


kdtree = KDTree(data)

kdtree
