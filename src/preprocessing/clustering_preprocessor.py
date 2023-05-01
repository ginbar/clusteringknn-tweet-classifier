import numpy as np
from numpy import ndarray
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

    def __init__(self, n_clusters:int, transforms:TextTransforms):
        super(ClusteringPreprocessor, self).__init__()
        self._n_clusters = n_clusters
        self._transforms = transforms
        self._dataset = None
        self._tokenized_tweets = None



    def fit(self, vectorized_data: ndarray, lemmatized_data: ndarray) -> None:
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
        self._model = KMeans(n_clusters=self._n_clusters, n_init='auto')

        self._dataset = lemmatized_data
        self._tokenized_tweets = vectorized_data

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
            
            cluster_mask = self._model.labels_==cluster_index
            lemmatized_data = self._dataset[cluster_mask]
            vectorized_data = self._tokenized_tweets[cluster_mask]

            if lemmatized_data.size:
                
                cluster_text = ' '.join(lemmatized_data)
                centroid = self._model.cluster_centers_[cluster_index]
                dissimilarities = np.array([np.linalg.norm(entry - centroid) for entry in vectorized_data])
                most_dissimilar = lemmatized_data[dissimilarities.argmax()]
                lesser_dissimilar = lemmatized_data[dissimilarities.argmin()]

                cluster = BottomLevelCluster(
                    cluster_index, 
                    lemmatized_data, 
                    None, 
                    centroid, 
                    cluster_text, 
                    lesser_dissimilar, 
                    most_dissimilar
                )
                
                clusters.append(cluster)
        
        return clusters



    def get_clustering_mask(self):
        return self._model.labels_