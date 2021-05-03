import numpy as np
import collections

from sklearn.base import ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from scipy.spatial import KDTree
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# downloading nltk corpuses files
nltk.download('stopwords')



class ClusterTreeKNN(ClassifierMixin):
    """docstring for ClusteringKNN."""

    def __init__(self):
        super(ClusterTreeKNN, self).__init__()
        self._initial_H_level_thereshould = 0.5
        self._kdtree = None



    def fit(self, data, clusters_masks, labels):
        
        self._kdtree = KDTree(data)

        B = clusters
        H = clusters[:,-1]
        B = clusters[:,-1]




    def predict(self, sample):
        pass




class ClusteringPreprocessor(object):
    """docstring for Preprocessor."""

    def __init__(self, n_clusters):
        super(ClusteringPreprocessor, self).__init__()
        self._model = KMeans(n_clusters=n_clusters)

    def create_clusters(self, data, n_clusters, labels):
        #TODO Remove this implementation
        clusters = []
        for label in range(n_clusters):
            cluster_data = data[labels==label]
            centroid = cluster_data.mean(axis=0)
            dissimilarities = cluster_data - centroid
            dissimilarities = np.abs(dissimilarities).sum(axis=1)
            ordering = np.argsort(dissimilarities, axis=0)
            cluster_data[ordering] = cluster_data
            clusters.append(cluster_data)
        return np.array(clusters)


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
