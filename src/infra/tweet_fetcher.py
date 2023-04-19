import json
import os
import tweepy
from config import HASHTAG_FILES_PATH
from preprocessing.text_transforms import TextTransforms


credentials = None
with open("twitter_credentials.json", "r") as file:
    credentials = json.load(file)



API_KEY = credentials['API_KEY']
API_SECRET_KEY = credentials['API_SECRET_KEY']



auth = tweepy.AppAuthHandler(API_KEY, API_SECRET_KEY)
api = tweepy.API(auth)
transforms = TextTransforms()



class TweetFetcher(object):

    def __init__(self):
        self._auth = tweepy.AppAuthHandler(API_KEY, API_SECRET_KEY)
        self._api = tweepy.API(auth)


    def get_cursor_for_hashtag(self, hashtag, max: int = 50) -> None:
        return tweepy.Cursor(self._api.search, q=hashtag).items(max)



def create_tweets_file_for_hashtag(hashtag: str, max: int = 50, path: str = None) -> None:
    """
    Creates a file containing tweets that contain a specified hashtag.
    
    Parameters
    ----------
        hashtag : str 
            A string that represents the hashtag to search for.
        max : int 
            An integer that represents the maximum number of tweets to retrieve (default is 50).
        path : str 
            A string that represents the file path where the tweets will be saved.

    Returns
    ----------
        None : This function does not return anything. Instead, it saves the tweets in a file specified by the path parameter.
    """
    
    if path is None:
        if not os.path.exists(HASHTAG_FILES_PATH):
            os.makedirs(HASHTAG_FILES_PATH)
        
        path = HASHTAG_FILES_PATH + hashtag + '/'
    
        if not os.path.exists(path):
            os.makedirs(path)
    
    raw_tweets_file_path = f'{path}/{hashtag}'
    lemmatized_tweets_file_path = f'{path}/{hashtag}.lem'

    with open(raw_tweets_file_path, 'a+') as raw_tweets_file, open(lemmatized_tweets_file_path, 'a+') as lemmatized_tweets_file:
        
        tweets = tweepy.Cursor(api.search, q=hashtag).items(max)
        
        for tweet in tweets:
            raw = tweet.text.replace('\n', ' ') + '\n'
            lemmatized = transforms.lemmatize(raw)
            
            if lemmatized:
                raw_tweets_file.write(raw)
                lemmatized_tweets_file.write(lemmatized)

