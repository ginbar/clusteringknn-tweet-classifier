import numpy as np
from dtos.preprocessing_results import PreprocessingResults
from infra.utils import create_file_path, create_dataset_folder



def save_preprocessing_results(hashtag: str, preprocessing: PreprocessingResults) -> None:
    
    centroids = np.array(preprocessing.centroids)
    
    create_dataset_folder(hashtag)
    
    np.save(create_file_path(hashtag, 'preprocessing', 'centr'), centroids)
    np.save(create_file_path(hashtag, 'preprocessing', 'mask'), preprocessing.clustering_mask)
    np.save(create_file_path(hashtag, 'preprocessing', 'assig'), preprocessing.assigned_labels)



def read_preprocessing_results(hashtag: str) -> PreprocessingResults:
    
    centroids = np.load(create_file_path(hashtag, 'preprocessing', 'centr'))
    clustering_mask = np.load(create_file_path(hashtag, 'preprocessing', 'mask'))
    assigned_labels = np.load(create_file_path(hashtag, 'preprocessing', 'assig'))

    return PreprocessingResults(centroids, clustering_mask, assigned_labels)
