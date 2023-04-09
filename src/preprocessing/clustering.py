import numpy as np
from sklearn.cluster import KMeans
from preprocessing.text_transforms import TextTransforms
from dtos.bottom_level_cluster import BottomLevelCluster




class ClusteringPreprocessor(object):
    """
    Generates the bottom layer clusters to be labeled.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters to be generated.

    language : str
        Language of the data to be inputed.
    """

    def __init__(self, n_clusters:int, language:str='portuguese'):
        super(ClusteringPreprocessor, self).__init__()
        self._language = language
        self._model = KMeans(n_clusters=n_clusters)
        self._transforms = TextTransforms(language)
        self._dataset = None
        self._tokenized_tweets = None



    def fit(self, data: list[str]) -> None:
        self._dataset = data
        tokenized_tweets = self._transforms.vectorize(data)
        print(tokenized_tweets)
        self._model.fit(tokenized_tweets)



    def create_clusters(self) -> list[BottomLevelCluster]:

        clusters = []

        for cluster_index in range(0, len(self._model.cluster_centers_)):
            cluster_data = self._dataset[self._model.labels_==cluster_index]
            cluster_text = ' '.join(cluster_data)

            centroid = self._model.cluster_centers_[cluster_index]

            cluster = BottomLevelCluster(cluster_index, cluster_text, "", "")
            
            clusters.append(cluster)
        
        return clusters

