from numpy import ndarray
from dataclasses import astuple, dataclass
from dtos.bottom_level_cluster import BottomLevelCluster


@dataclass
class PreprocessingResults:
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
    centroids:ndarray
    clustering_mask:ndarray
    assigned_labels:ndarray
    