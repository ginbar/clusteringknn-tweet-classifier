import numpy as np
from numpy import ndarray
from infra.utils import create_file_path, create_dataset_folder



def save_results(hashtag: str, results: ndarray) -> None:
    
    create_dataset_folder(hashtag)
    
    np.save(create_file_path(hashtag, None, 'result'), results)



def read_results(hashtag: str):
    return np.load(create_file_path(hashtag, None, 'result'))



def read_ground_truth(hashtag: str):
    return np.load(create_file_path(hashtag, None, 'ground_truth'))