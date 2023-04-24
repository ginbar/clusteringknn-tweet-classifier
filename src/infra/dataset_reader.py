from infra.utils import create_file_path, create_dataset_folder


class DatasetReader(object):
    
    def __init__(self, hashtag:str, folder:str):
        self._hashtag = hashtag
        self._raw_tweets_file_path = create_file_path(hashtag, folder, 'raw')
        self._lemmatized_tweets_file_path = create_file_path(hashtag, folder, 'lemm') 
        self._vectorized_tweets_file_path = create_file_path(hashtag, folder, 'vect')
        self._raw_tweets_file = None
        self._lemmatized_tweets_file = None



    def get_raw_tweets(self):
        with open(self._raw_tweets_file_path, "r") as raw_tweets_file:
            return raw_tweets_file.readlines()



    def get_lemmatized_tweets(self):
        with open(self._lemmatized_tweets_file_path, "r") as lemmatized_tweets_file:
            return lemmatized_tweets_file.readlines()



    def get_vectorized_tweets(self):
        with open(self._vectorized_tweets_file_path, "r") as vectorized_tweets_file:
            return vectorized_tweets_file.readlines()



    def __enter__(self):
        create_dataset_folder(self._hashtag)
        return self



    def __exit__(self, exc_type, exc_value, traceback):
        self._raw_tweets_file.close()
        self._lemmatized_tweets_file.close()