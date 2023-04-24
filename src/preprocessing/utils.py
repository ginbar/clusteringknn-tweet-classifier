from numpy import ndarray
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from dtos.preprocessing_results import PreprocessingResults


def silhouette_method(data:ndarray, min_n_clusters:int=3, max_n_clusters:int=15):
    coefficients = []
    
    for n_cluster in range(min_n_clusters, max_n_clusters):
        kmeans = KMeans(n_clusters=n_cluster).fit(data)
        labels = kmeans.labels_
        coefficient = silhouette_score(data, labels, metric='euclidean')
        coefficients.append(coefficient)

    return coefficients.index(max(coefficients)) + min_n_clusters




def remove_invalid_clusters(data:ndarray, preprocessing_results:PreprocessingResults):
    assigned_labels = preprocessing_results.assigned_labels
    
    invalid_clusters_indexes = []

    for index, (most_dis_label, lesser_dis_label) in enumerate(assigned_labels):
        if most_dis_label == lesser_dis_label:
            invalid_clusters_indexes.append(index)
    
    clustering_mask = preprocessing_results.clustering_mask
    invalid_mask = invalid_clusters_indexes == clustering_mask

    cleaned_data = data[invalid_mask]
    clenead_mask = preprocessing_results.clustering_mask[invalid_mask]
    cleaned_centroids = preprocessing_results.centroids[invalid_mask]

    return cleaned_data, PreprocessingResults(cleaned_centroids, clenead_mask, cleaned_centroids)