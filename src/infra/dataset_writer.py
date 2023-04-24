import os
from infra.utils import create_file_path, create_dataset_folder


class DatasetWriter(object):
    

    def __init__(self, hashtag:str, folder:str):
        self._hashtag = hashtag
        self._raw_tweets_file_path = create_file_path(hashtag, folder)
        self._lemmatized_tweets_file_path = create_file_path(hashtag, folder, 'lemm')
        self._raw_tweets_file = None
        self._lemmatized_tweets_file = None


    
    def save_raw_tweet(self, tweet):
        self._raw_tweets_file.write(tweet)



    def save_lemmatized_tweet(self, tweet):
        self._lemmatized_tweets_file.write(tweet)
       

    
    def __enter__(self):
        create_dataset_folder(self._hashtag)
        
        os.umask(0)
        flags = os.O_CREAT | os.O_WRONLY
        mode = 0o777

        self._raw_tweets_file = open(os.open(self._raw_tweets_file_path, flags, mode) , 'w')
        self._lemmatized_tweets_file = open(os.open(self._lemmatized_tweets_file_path, flags, mode), 'w')

        return self



    def __exit__(self, exc_type, exc_value, traceback):
        self._raw_tweets_file.close()
        self._lemmatized_tweets_file.close()
