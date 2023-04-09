
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
    def __init__(
        self, 
        index:int, 
        text:str, 
        lesser_dissimilar:str, 
        most_dissimilar:str
    ):
        self.index = index
        self.name = 'Cluster ' + str(index)
        self.text = text 
        self.most_dissimilar = most_dissimilar
        self.lesser_dissimilar = lesser_dissimilar
