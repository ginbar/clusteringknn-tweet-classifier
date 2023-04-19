import os
from config import HASHTAG_FILES_PATH



class DatasetWriter(object):
    
    def __init__(self, hashtag:str, folder:str):
        self._hastag = hashtag
        self._raw_tweets_file_path = f'{HASHTAG_FILES_PATH}/{hashtag}/{folder}/{hashtag}.raw'
        self._lemmatized_tweets_file_path = f'{HASHTAG_FILES_PATH}/{hashtag}/{folder}/{hashtag}.lemm'
        self._raw_tweets_file = None
        self._lemmatized_tweets_file = None


    
    def save_raw_tweet(self, tweet):
        self._raw_tweets_file.write(tweet)



    def save_lemmatized_tweet(self, tweet):
        self._lemmatized_tweets_file.write(tweet)
       

    
    def __enter__(self):

        if not os.path.exists(HASHTAG_FILES_PATH):
            os.makedirs(HASHTAG_FILES_PATH)
        
        path = HASHTAG_FILES_PATH + self._hashtag + '/'
    
        if not os.path.exists(path):
            os.makedirs(path)

        self._raw_tweets_file = open(self._raw_tweets_file_path, "a+")
        self._lemmatized_tweets_file = open(self._lemmatized_tweets_file_path, "a+")

        return self



    def __exit__(self, exc_type, exc_value, traceback):
        self._raw_tweets_file.close()
        self._lemmatized_tweets_file.close()