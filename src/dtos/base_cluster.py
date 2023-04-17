from numpy import ndarray
from dtos.bottom_level_cluster import BottomLevelCluster

class BaseCluster(object):
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
        index:int, 
        data:ndarray,
        children:list[BottomLevelCluster]=None
    ):
        self.index = index
        self.name = 'Cluster ' + str(index)
        self.data = data 
        self.children = children


    def add_child(self, child:BottomLevelCluster) -> None:
        self.children.append(child)