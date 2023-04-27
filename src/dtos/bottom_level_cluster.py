from numpy import ndarray
from dataclasses import astuple, dataclass


@dataclass
class BottomLevelCluster(object):
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
    index:int
    data:ndarray
    labels:ndarray
    centroid:ndarray
    text:str=None
    lesser_dissimilar:str=None
    most_dissimilar:str=None


    @property
    def name(self):
        return 'Cluster ' + str(self.index)
