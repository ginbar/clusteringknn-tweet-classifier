import json
import os
import tweepy



credentials = None
with open("twitter_credentials.json", "r") as file:
    credentials = json.load(file)



HASHTAG_FILES_PATH = 'data/hastags/' 
API_KEY = credentials['API_KEY']
API_SECRET_KEY = credentials['API_SECRET_KEY']



auth = tweepy.AppAuthHandler(API_KEY, API_SECRET_KEY)
api = tweepy.API(auth)



def create_tweets_file_for_hashtag(hashtag, max=50, path=None):
    
    if path is None:
        if not os.path.exists(HASHTAG_FILES_PATH):
            os.makedirs(HASHTAG_FILES_PATH)
        path = HASHTAG_FILES_PATH + hashtag
    
    with open(path, 'a+') as f:
        tweets = tweepy.Cursor(api.search, q=hashtag).items(max)
        for tweet in tweets:
            f.write(tweet.text + '\n')

