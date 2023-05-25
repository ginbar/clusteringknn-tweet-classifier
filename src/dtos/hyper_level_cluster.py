from numpy import ndarray
from dtos.bottom_level_cluster import BottomLevelCluster
from dataclasses import astuple, dataclass


@dataclass
class HyperLevelCluster(object):
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
    label:any
    gamma_d:ndarray
    children:list[BottomLevelCluster]=None


    def add_child(self, child:BottomLevelCluster) -> None:
        if self.children is None:
            self.children = []
        self.children.append(child)
