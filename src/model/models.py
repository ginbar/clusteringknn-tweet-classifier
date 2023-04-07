import numpy as np
import collections

from sklearn.base import ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from scipy.cluster.hierarchy import dendrogram, linkage, ClusterNode
from scipy.spatial.distance import euclidean
from scipy.spatial import KDTree


class ClusterTreeKNN(ClassifierMixin):
    """
    docstring for ClusteringKNN.
    """

    def __init__(self):
        super(ClusterTreeKNN, self).__init__()

        # Parameters
        self._initial_H_level_thereshould = 0.5
        
        self._kdtree = None
        self._Blevel = None



    def fit(self, data, clusters_masks, centroids, c_distances, valid_clusters, labels):
        
        self._kdtree = KDTree(data)

        self._Blevel = clusters
        
        for vc in valid_clusters:
            cluster = data[clusters_masks == vc]



        # most_dissimilars =  

        H = clusters[:,-1]
        B = clusters[:,-1]




    def predict(self, sample):
        pass
