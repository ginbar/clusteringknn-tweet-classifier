import os

HASHTAG_FILES_PATH = 'data/datasets/hashtags'



def create_dataset_folder(hashtag):
    
    if not os.path.exists(HASHTAG_FILES_PATH):
        os.makedirs(HASHTAG_FILES_PATH)
        
    path = f'{HASHTAG_FILES_PATH}/{hashtag}/'

    if not os.path.exists(path):
        os.makedirs(path)

    train_dataset_file = path + '/train'
    test_dataset_file = path + '/test'
    preprocessing_dataset_file = path + '/preprocessing'

    if not os.path.exists(train_dataset_file):
        os.makedirs(train_dataset_file)

    if not os.path.exists(test_dataset_file):
        os.makedirs(test_dataset_file)

    if not os.path.exists(preprocessing_dataset_file):
        os.makedirs(preprocessing_dataset_file)



def create_file_path(hashtag, folder=None, suffix=None):
    path = f'{HASHTAG_FILES_PATH}/{hashtag}'
    
    if not folder is None:
        path += f'/{folder}/{hashtag}'
    
    if not suffix is None:
        path += f'.{suffix}'
    
    return f'{HASHTAG_FILES_PATH}/{hashtag}/{folder}/{hashtag}.{suffix}'