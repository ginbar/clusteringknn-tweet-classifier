import numpy as np
from numpy import ndarray
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from dtos.preprocessing_results import PreprocessingResults



def silhouette_method(data:ndarray, min_n_clusters:int=3, max_n_clusters:int=15):
    coefficients = []
    
    for n_cluster in range(min_n_clusters, max_n_clusters + 1):
        kmeans = KMeans(n_clusters=n_cluster, n_init='auto').fit(data)
        labels = kmeans.labels_
        coefficient = silhouette_score(data, labels)
        coefficients.append(coefficient)

    return coefficients.index(max(coefficients)) + min_n_clusters



def remove_invalid_clusters(data:ndarray, preprocessing_results:PreprocessingResults):
    assigned_labels = preprocessing_results.assigned_labels
    
    invalid_clusters_indexes = []

    for index, (most_dis_label, lesser_dis_label) in enumerate(assigned_labels.T):
        if most_dis_label != lesser_dis_label:
            invalid_clusters_indexes.append(index)
    
    centroids = preprocessing_results.centroids
    clustering_mask = preprocessing_results.clustering_mask
    assigned_labels = preprocessing_results.assigned_labels

    invalid_mask = clustering_mask == invalid_clusters_indexes

    if len(invalid_clusters_indexes) > 0:

        invalid_mask = clustering_mask == invalid_clusters_indexes
        
        data = data[invalid_mask]
        centroids = centroids[invalid_mask]
        clustering_mask = clustering_mask[invalid_mask]
        assigned_labels = assigned_labels[invalid_mask]
        
    return data, PreprocessingResults(centroids, clustering_mask, assigned_labels)
    


def create_labeling(preprocessing:PreprocessingResults):
    y = np.zeros(preprocessing.clustering_mask.shape[0])

    for index, (label, _) in enumerate(preprocessing.assigned_labels.T):
        np.putmask(y, preprocessing.clustering_mask == index, label)
    
    return y