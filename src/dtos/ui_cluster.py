from numpy import ndarray

class UICluster(object):
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
        centroid:ndarray,
        text:str=None, 
        lesser_dissimilar:str=None, 
        most_dissimilar:str=None
    ):
        self.index = index
        self.name = 'Cluster ' + str(index)
        self.data = data
        self.centroid = centroid
        self.text = text 
        self.most_dissimilar = most_dissimilar
        self.lesser_dissimilar = lesser_dissimilar
        
