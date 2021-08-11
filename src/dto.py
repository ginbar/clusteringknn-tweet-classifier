
class Cluster(object):
    
    def __init__(self, name, text, lesser_dissimilar, most_dissimilar):
        self.name = name
        self.text = text 
        self.most_dissimilar = most_dissimilar
        self.lesser_dissimilar = lesser_dissimilar
