import os

HASHTAG_FILES_PATH = 'data/datasets/hashtags'

def create_dataset_folder(hashtag):
    
    if not os.path.exists(HASHTAG_FILES_PATH):
        os.makedirs(HASHTAG_FILES_PATH)
        
    path = HASHTAG_FILES_PATH + hashtag + '/'

    if not os.path.exists(path):
        os.makedirs(path)


def create_file_path(hashtag, folder, suffix):
    return f'{HASHTAG_FILES_PATH}/{hashtag}/{folder}/{hashtag}.{suffix}'