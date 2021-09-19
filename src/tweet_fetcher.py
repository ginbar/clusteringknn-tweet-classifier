import json
import os
import tweepy

from lemmatizer import lemmatize


credentials = None
with open("twitter_credentials.json", "r") as file:
    credentials = json.load(file)



HASHTAG_FILES_PATH = 'data/hashtags/' 
API_KEY = credentials['API_KEY']
API_SECRET_KEY = credentials['API_SECRET_KEY']



auth = tweepy.AppAuthHandler(API_KEY, API_SECRET_KEY)
api = tweepy.API(auth)



def create_tweets_file_for_hashtag(hashtag, max=50, path=None):
    
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

