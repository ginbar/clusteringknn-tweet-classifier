import json
import os
import tweepy

from preprocessing.lemmatizer import lemmatize


credentials = None
with open("twitter_credentials.json", "r") as file:
    credentials = json.load(file)



HASHTAG_FILES_PATH = 'data/hashtags/' 
API_KEY = credentials['API_KEY']
API_SECRET_KEY = credentials['API_SECRET_KEY']



auth = tweepy.AppAuthHandler(API_KEY, API_SECRET_KEY)
api = tweepy.API(auth)



def create_tweets_file_for_hashtag(hashtag: str, max: int = 50, path: str = None) -> None:
    """
    Creates a file containing tweets that contain a specified hashtag.
    
    Parameters:
    - hashtag (str): A string that represents the hashtag to search for.
    - max (int): An integer that represents the maximum number of tweets to retrieve (default is 50).
    - path (str): A string that represents the file path where the tweets will be saved.

    Returns:
    - None: This function does not return anything. Instead, it saves the tweets in a file specified by the path parameter.
    """
    
    if path is None:
        if not os.path.exists(HASHTAG_FILES_PATH):
            os.makedirs(HASHTAG_FILES_PATH)
        
        path = HASHTAG_FILES_PATH + hashtag + '/'
    
        if not os.path.exists(path):
            os.makedirs(path)
    
    rfile_path = f'{path}/{hashtag}'
    lfile_path = f'{path}/{hashtag}.lem'

    with open(rfile_path, 'a+') as rfile, open(lfile_path, 'a+') as lfile:
        
        tweets = tweepy.Cursor(api.search, q=hashtag).items(max)
        
        for tweet in tweets:
            inline = tweet.text.replace('\n', ' ') + '\n'
            lemmatized = lemmatize(inline)
            
            if lemmatized:
                rfile.write(inline)
                lfile.write(lemmatized)

