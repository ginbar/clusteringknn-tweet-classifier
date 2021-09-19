import numpy as np
import collections

from sklearn.base import ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, ClusterNode
from scipy.spatial.distance import euclidean
from scipy.spatial import KDTree
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords



class ClusterTreeKNN(ClassifierMixin):
    """docstring for ClusteringKNN."""

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



        most_dissimilars =  

        H = clusters[:,-1]
        B = clusters[:,-1]




    def predict(self, sample):
        pass




class ClusteringPreprocessor(object):
    """docstring for Preprocessor."""

    def __init__(self, n_clusters):
        super(ClusteringPreprocessor, self).__init__()
        self._model = KMeans(n_clusters=n_clusters)

    def create_clusters_masks(self, data):
        tokenized_tweets = self.tokenizer(data)
        self._model.fit(tokenized_tweets)
        return self._model.labels_


    def tokenizer(tweets):
        #TODO Implement multiple languages  
        #TODO Try to improve performance
        #TODO Apply lematization
        langsw = stopwords.words('portuguese')
        return np.array(w for w in word_tokenize(t.lower()) if w.isalpha() and not w in langsw for t in tweets)
