
class Cluster(object):
    """
    Representes a cluster created at the first phase of the method.
    """
    def __init__(self, index, text, lesser_dissimilar, most_dissimilar):
        self.index = index
        self.name = 'Cluster ' + index
        self.text = text 
        self.most_dissimilar = most_dissimilar
        self.lesser_dissimilar = lesser_dissimilar
