
class Cluster(object):
    
    def __init__(self, name, text, most_dissimilar, lesser_dissimilar):
        self.name = name
        self.text = text 
        self.most_dissimilar = most_dissimilar
        self.lesser_dissimilar = lesser_dissimilar
