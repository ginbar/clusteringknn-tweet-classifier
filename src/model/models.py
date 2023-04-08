import numpy as np
import collections

from sklearn.base import ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from scipy.cluster.hierarchy import dendrogram, linkage, ClusterNode
from scipy.spatial.distance import euclidean
from scipy.spatial import KDTree


class ClusterTreeKNN(ClassifierMixin):
    """
    KNN over a cluster based tree.
    
    Parameters
    ----------
    initial_H_level_thereshould : float, default=0.5
        The maximum distance between two samples for one to be considered
        as in the neighborhood of th

    References
    ----------
    Oliveira, E., Roatti, H., Nogueira, M., Basoni, H. e Ciarelli M. (2015). Using the Cluster-based
    Tree Structure of k-Nearest Neighbor to Reduce the Effort Required to Classify Unlabeled Large
    Datasets.

    Zhang, B. e Srihari, S. N. (2004). Fast k-Nearest Neighbor Classification Using Cluster-Based
    Trees. IEEE Trans. Pattern Anal. Mach. Intell., 26(4): 525–528.
    """

    def __init__(
        self, 
        initial_H_level_thereshould= 0.5
    ):
        super(ClusterTreeKNN, self).__init__()

        # Parameters
        self._initial_H_level_thereshould = initial_H_level_thereshould
        
        self._Blevel = None



    def fit(self, data, clusters_masks, centroids, c_distances, valid_clusters, labels):

        # Step 1. Initialize the bottom level of the cluster tree
        # with all template documents that are labeled
        # during the process described in Section 4.1.
        # These templates constitute a single level B ;
        self._Blevel = clusters
        
        # Step 2. ∀Sl ∈ Ω, extract one of the most dissimilar
        # samples, for instance d1 , and compute the lo-
        # cal properties of each sample d1 = d: γ(d),
        # ψ(d) and ℓ(d). Then, rank all clusters Sl in
        # descending order of ℓ(.);
        
        # Step 3. Take the sample d1 with the biggest ℓ(.) as
        # a hypernode, and let all samples of ψ(d1 ) be
        # nodes at the bottom level of the tree, B . Then,
        # remove d1 and all samples in ψ(d1 ) from the
        # original dataset, and set up a link between d1
        # and each pattern of ψ(d1 ) in B ;
        
        # Step 4. Repeat Step 2 and Step 3 until the Ω set be-
        # comes empty. At this point, the cluster tree is
        # configured with a hyperlevel, H , and a bot-
        # tom level, B ;
        
        # Step 5. Select a threshold η and cluster all templates
        # in H so that the radius of each cluster is less
        # than or equal to η. All cluster centers form
        # another level of the cluster tree, P ;
        # Step 6. Increase the threshold η and repeat Step 5 for
        # all nodes at the level P until a single node is
        # left in the resulting level.


    def predict(self, sample):
        pass
        
        # Step 1. First, we compute the dissimilarity between x
        # and each node at the top level of the cluster
        # tree and choose the ς nearest nodes as a node
        # set Lx ;
        
        # Step 2. Compute the dissimilarity between x and
        # each subnode linked to the nodes in Lx , and
        # again choose the ς nearest nodes, which are
        # used to update the node set Lx ;
        
        # Step 3. Repeat Step 2 until reaching the hyperlevel
        # in the tree. When the searching stops at the
        # hyperlevel, Lx consists of ς hypernodes;
        
        # Step 4. Search Lx for the hypernode:
        # Lh = {Y |d(y, x) ≤ γ(d), y ∈ Lx }. If all nodes in Lh
        # have the same class label, then this class is as-
        # sociated with x and the classification process
        # stops; otherwise, go to Step 5;
        
        # Step 5. Compute the dissimilarity between x and ev-
        # ery subnode linked to the nodes in Lx , and
        # choose the k nearest samples. Then, take a
        # majority voting among the k nearest samples
        # to determine the class label for x.