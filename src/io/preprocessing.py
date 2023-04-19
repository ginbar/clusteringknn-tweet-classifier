import numpy as np
from dtos.preprocessing_results import PreprocessingResults
from config import HASHTAG_FILES_PATH



def save_preprocessing_results(hashtag: str, preprocessing: PreprocessingResults) -> None:
    
    centroids = np.array(preprocessing.centroids)
    
    np.save(f'{HASHTAG_FILES_PATH}/{hashtag}.centr', centroids)
    np.save(f'{HASHTAG_FILES_PATH}/{hashtag}.mask', preprocessing.clustering_mask)
    np.save(f'{HASHTAG_FILES_PATH}/{hashtag}.assig', preprocessing.assigned_labels)



def read_preprocessing_results(hashtag: str) -> PreprocessingResults:
    
    centroids = np.load(f'{HASHTAG_FILES_PATH}/{hashtag}.centr')
    clustering_mask = np.load(f'{HASHTAG_FILES_PATH}/{hashtag}.mask')
    assigned_labels = np.load(f'{HASHTAG_FILES_PATH}/{hashtag}.assig')

    return PreprocessingResults(centroids, clustering_mask, assigned_labels)
