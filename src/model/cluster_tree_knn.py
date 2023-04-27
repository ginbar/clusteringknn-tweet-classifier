import numpy as np
from numpy import ndarray
from sklearn.base import ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage, ClusterNode
from scipy.spatial import distance
from scipy.spatial import KDTree
import scipy

from dtos.bottom_level_cluster import BottomLevelCluster
from dtos.hyper_level_cluster import HyperLevelCluster 
from dtos.upper_level_cluster import UpperLevelCluster 



class ClusterTreeKNN(ClassifierMixin):
    """
    KNN over a cluster based tree.
    
    Parameters
    ----------
    initial_hyperlevel_threshold : float, default=0.5
        Tthreshold for the radius of each cluster at the hypernode level.
    
    sigma_nearest_nodes : int, default=5
        Number of nodes be searched in the upper level at prediction.

    References
    ----------
    Oliveira, E., Roatti, H., Nogueira, M., Basoni, H. e Ciarelli M. (2015). Using the Cluster-based
    Tree Structure of k-Nearest Neighbor to Reduce the Effort Required to Classify Unlabeled Large
    Datasets.

    Zhang, B. e Srihari, S. N. (2004). Fast k-Nearest Neighbor Classification Using Cluster-Based
    Trees. IEEE Trans. Pattern Anal. Mach. Intell., 26(4): 525-528.
    """

    def __init__(
        self,
        n_neighbors:int=5, 
        metric:callable=distance.euclidean,
        initial_hyperlevel_threshold:float=0.5,
        sigma_nearest_nodes:int=5
    ):
        super(ClusterTreeKNN, self).__init__()

        # Parameters
        self._n_neighbors = n_neighbors
        self._metric = metric
        self._initial_hyperlevel_threshold = initial_hyperlevel_threshold
        self._sigma_nearest_nodes = sigma_nearest_nodes

        self._Blevel = None
        self._Hlevel = None
        self._Plevel = None



    def fit(
        self, 
        X:ndarray,
        y:ndarray, 
        clusters_masks:ndarray[int],
        centroids:ndarray,
    ) -> None:
        """
        Fit the clustering k-nearest neighbors classifier from the training dataset. 
        It assumes that just valid clusters are passed as arguments.
        
        Parameters
        ----------
        X : ndarray
            Training data.

        y : ndarray
            Target values.
        
        Returns
        ----------
            self : The fitted clustering k-nearest neighbors classifier.
        """
        
        # Step 1. 
        # Initialize the bottom level of the cluster tree
        # with all template documents that are labeled
        # during the process described in Section 4.1.
        # These templates constitute a single level B;
        n_clusters = len(np.unique(clusters_masks))
        self._Blevel = [BottomLevelCluster(i, X[i == clusters_masks], centroids[i]) for i in range(n_clusters)]
        self._Hlevel = []
        self._Plevel = []

        X_length = X.shape[0]

        remaining_clusters = [i for i in range(n_clusters)]
        
        while len(remaining_clusters) > 0:
            
            # Step 2. 
            # ∀Sl ∈ Ω, extract one of the most dissimilar
            # samples, for instance d1 , and compute the lo-
            # cal properties of each sample d1 = d: γ(d),
            # ψ(d) and ℓ(d). Then, rank all clusters Sl in
            # descending order of ℓ(.);
            
            remaining_clusters_mask = np.isin(clusters_masks, remaining_clusters)

            _clusters_masks = clusters_masks[remaining_clusters_mask]
            _X = X[remaining_clusters_mask]
            _y = y[remaining_clusters_mask]

            X_length = _X.shape[0]
            most_dissimilars = np.array([self._Blevel[c].data[np.linalg.norm(self._Blevel[c].data - centroids[c], axis=1).argmax()] for c in remaining_clusters])

            most_dissimilars_indexes = np.array([np.where((_X == m).all(axis=1))[0][0] for m in most_dissimilars])
            dissimilarities = np.array([np.linalg.norm(i - j) for i in _X for j in _X]).reshape((X_length, X_length))
            
            if np.unique(_y).shape[0] > 1:
                gamma_d = np.array([dissimilarities[i][_y[i] != _y].min() for i in most_dissimilars_indexes])
                psi_d = [_X[(_y[i] == _y) & (dissimilarities[i] < gamma_d[k])] for k, i in enumerate(most_dissimilars_indexes)]
                lambda_d = np.array([len(psi) for psi in psi_d])
                new_hyper_node_index = lambda_d.argmax()
                
                # Step 3. 
                # Take the sample d1 with the biggest ℓ(.) as
                # a hypernode, and let all samples of ψ(d1 ) be
                # nodes at the bottom level of the tree, B . Then,
                # remove d1 and all samples in ψ(d1 ) from the
                # original dataset, and set up a link between d1
                # and each pattern of ψ(d1 ) in B ;

                new_hyper_node = HyperLevelCluster(len(self._Hlevel), most_dissimilars[new_hyper_node_index], _y[most_dissimilars_indexes[new_hyper_node_index]])

                self._Hlevel.append(new_hyper_node)
                
                new_bottom_level_node = BottomLevelCluster(len(self._Blevel), psi_d[new_hyper_node_index], None)
                cluster_to_be_removed = _clusters_masks[most_dissimilars_indexes[new_hyper_node_index]]
                
                remaining_clusters.remove(cluster_to_be_removed)
                new_hyper_node.add_child(new_bottom_level_node)
                
                self._Blevel.append(new_bottom_level_node)
            else:
                remaining_clusters.pop()

            # Step 4. 
            # Repeat Step 2 and Step 3 until the Ω set be-
            # comes empty. At this point, the cluster tree is
            # configured with a hyperlevel, H , and a bottom level, B ;
        
        # Step 5. 
        # Select a threshold η and cluster all templates
        # in H so that the radius of each cluster is less
        # than or equal to η. All cluster centers form
        # another level of the cluster tree, P ;
        hyperlevel_threshold = self._initial_hyperlevel_threshold

        self._Plevel = self._Hlevel

        while len(self._Plevel) > 1:
            hyperlevel_data = np.array([cluster.data for cluster in self._Plevel]) 
            new_P_level = None
            
            if hyperlevel_data.shape[0] == 2:

                hyperlevel_clustering = DBSCAN(eps=self._initial_hyperlevel_threshold, min_samples=2)
                hyperlevel_clustering.fit(hyperlevel_data)
                
                new_P_level = [UpperLevelCluster(i, hyperlevel_data[centr_ind]) for i, centr_ind in enumerate(np.unique(hyperlevel_clustering.core_sample_indices_))]
            else:
                new_P_level = [UpperLevelCluster(i, hyperlevel_data[i]) for i in range(2)]

            self._Plevel = new_P_level
            hyperlevel_threshold = hyperlevel_threshold + 0.5
            # Step 6. 
            # Increase the threshold η and repeat Step 5 for
            # all nodes at the level P until a single node is
            # left in the resulting level.

        return self



    def predict(self, sample: ndarray) -> ndarray:
        if len(sample.shape) > 1:
            return np.array([self._predict(_sample) for _sample in sample])
        return self._predict(sample)



    def _predict(self, sample: ndarray) -> ndarray:
        """
        Predict the class labels for the provided data.
        
        Parameters
        ----------
        sample : ndarray
            Test samples.

        Returns
        ----------
            ndarray : Class labels for each data sample.
        """
        
        # Step 1. 
        # First, we compute the dissimilarity between x
        # and each node at the top level of the cluster
        # tree and choose the ς nearest nodes as a node
        # set Lx ;
        distances = np.array([cluster.data - sample for cluster in self._Plevel])
        indexes = distances.argsort()[:self._sigma_nearest_nodes]
        
        Lx = np.take(self._Plevel, indexes)
        print(Lx)
        while all(not cluster is HyperLevelCluster for cluster in Lx):

            # Step 2. 
            # Compute the dissimilarity between x and
            # each subnode linked to the nodes in Lx , and
            # again choose the ς nearest nodes, which are
            # used to update the node set Lx ;
            subnodes = np.array([cluster.children for cluster in self._Plevel]).flatten()
            
            distances = np.array([cluster.data - sample for cluster in subnodes])
            indexes = distances.argsort()[:self._sigma_nearest_nodes]
            
            Lx = np.take(self._Plevel, indexes)

            # Step 3. 
            # Repeat Step 2 until reaching the hyperlevel
            # in the tree. When the searching stops at the
            # hyperlevel, Lx consists of ς hypernodes;

        # Step 4. 
        # Search Lx for the hypernode:
        # Lh = {Y |d(y, x) ≤ γ(d), y ∈ Lx }. 
        Lh_hyper_nodes = None #TODO ????
        
        # If all nodes in Lh have the same class label, then this class is as-
        # sociated with x and the classification process
        # stops; otherwise, go to Step 5;
        Lh_hyper_node_labels = np.unique(np.array([c.label for c in Lh_hyper_nodes]))
        
        if not len(Lh_hyper_node_labels) > 1:
            return Lh_hyper_node_labels[0]
        else:
            # Step 5. 
            # Compute the dissimilarity between x and
            # every subnode linked to the nodes in Lx , and
            # choose the k nearest samples. Then, take a
            # majority voting among the k nearest samples
            # to determine the class label for x.
            bottom_level_data = np.array([c.data for c in Lh_hyper_nodes])
            bottom_level_label = np.array([c.label for c in Lh_hyper_nodes])

            knn = KNeighborsClassifier(n_neighbors=self._n_neighbors)

            knn.fit(bottom_level_data, bottom_level_label)

            return knn.predict(sample)
            