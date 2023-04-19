from numpy import ndarray
from dtos.bottom_level_cluster import BottomLevelCluster

class PreprocessingResults(object):
    """
    Representes a cluster created at the first phase of the method.

    Parameters
    ----------
    index : int
        Index of the generated cluster.

    text : str
        All the text belonging to the cluster. Usefull for display a cloud of words.
    
    lesser_dissimilar : str
        Lesser dissimilar cluster entry(closest to centroid).
         
    most_dissimilar : str
        Most dissimilar cluster entry(farthest from centroid).
    """
    def __init__(
        self,
        centroids,
        clustering_mask,
        assigned_labels,
    ):
        self.centroids = centroids
        self.clustering_mask = clustering_mask
        self.assigned_labels = assigned_labels
