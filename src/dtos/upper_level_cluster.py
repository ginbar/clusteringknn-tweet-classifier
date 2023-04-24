from numpy import ndarray
from dtos.hyper_level_cluster import HyperLevelCluster
from dataclasses import astuple, dataclass


@dataclass
class UpperLevelCluster(object):
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
    children:list[HyperLevelCluster]=None


    def add_child(self, child:HyperLevelCluster) -> None:
        self.children.append(child)
        