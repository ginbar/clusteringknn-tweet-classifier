from config import HASHTAG_FILES_PATH



class DatasetReader(object):
    
    def __init__(self, hashtag:str, folder:str):
        self._hastag = hashtag
        self._raw_tweets_file_path = f'{HASHTAG_FILES_PATH}/{hashtag}/{folder}/{hashtag}.raw'
        self._lemmatized_tweets_file_path = f'{HASHTAG_FILES_PATH}/{hashtag}/{folder}/{hashtag}.lemm'
        self._vectorized_tweets_file_path = f'{HASHTAG_FILES_PATH}/{hashtag}/{folder}/{hashtag}.vect'
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
        return self



    def __exit__(self, exc_type, exc_value, traceback):
        self._raw_tweets_file.close()
        self._lemmatized_tweets_file.close()