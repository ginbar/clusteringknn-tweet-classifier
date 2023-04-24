import numpy as np
from numpy import ndarray
from sklearn.cluster import KMeans
from preprocessing.text_transforms import TextTransforms
from dtos.bottom_level_cluster import BottomLevelCluster
from preprocessing.utils import silhouette_method



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

    def __init__(self, n_clusters:int=None, language:str='portuguese'):
        super(ClusteringPreprocessor, self).__init__()
        self._n_clusters = n_clusters
        self._transforms = TextTransforms(language)
        self._dataset = None
        self._tokenized_tweets = None



    def fit(self, lemmatized_data: list[str]) -> None:
        """
        Generates the bottom layer clusters to be labeled.
        
        Parameters
        ----------
        data : list[str]
            Texts to be clusterized.

        Returns
        ----------
            None : The method returns nothing.
        """
        vectorized_data = self._transforms.vectorize(lemmatized_data)

        n_clusters = silhouette_method(vectorized_data) if self._n_clusters is None else self._n_clusters

        self._model = KMeans(n_clusters=n_clusters)

        self._dataset = np.array(lemmatized_data)
        self._model.fit(vectorized_data)



    def create_clusters(self) -> list[BottomLevelCluster]:
        """
        Generates the bottom layer clusters to be labeled.
        
        Parameters
        ----------
        n_clusters : int
            Number of clusters to be generated.

        language : str
            Language of the data to be inputed.
        
        Returns
        ----------
            list[BottomLevelCluster] : BottomLevelClusters with the lesser and most dissimilar entries.
        """
        clusters = []

        for cluster_index in range(len(self._model.cluster_centers_)):

            cluster_data = self._dataset[self._model.labels_==cluster_index]
            
            if cluster_data.size:
                
                cluster_text = ' '.join(cluster_data)
                centroid = self._model.cluster_centers_[cluster_index]
                
                cluster = BottomLevelCluster(cluster_index, cluster_data, centroid, cluster_text, "", "")
                
                clusters.append(cluster)
        
        return clusters



    def get_clustering_mask(self):
        return self._model.labels_