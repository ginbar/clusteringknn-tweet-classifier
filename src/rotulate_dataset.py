from ui.word_cloud_ui import WordCloudUI
from dtos.cluster import Cluster


text = 'Supervised learning is the machine learning task of learning a function that maps an input to \
        an output based on example input-output pairs.[1] It infers a function from labeled training data \
        consisting of a set of training examples.[2] In supervised learning, each example is a pair consisting \
        of an input object (typically a vector) and a desired output value (also called the supervisory signal). \
        A supervised learning algorithm analyzes the training data and produces an inferred function, which can \
        be used for mapping new examples. An optimal scenario will allow for the algorithm to correctly determine \
        the class labels for unseen instances. This requires the learning algorithm to generalize from the training \
        data to unseen situations in a "reasonable" way (see inductive bias).'

clusters = [
        Cluster('Cluster1', text, 'Some opinions about a subject.', 'Another opinion about the same subject.'),
        Cluster('Cluster2', text, 'ld2', 'md2'),
        Cluster('Cluster3', text, 'ld3', 'md3')]

gui = WordCloudUI(clusters=clusters)

#gui.feed_cluster_data('Cluster 1', text) 
gui.show()
