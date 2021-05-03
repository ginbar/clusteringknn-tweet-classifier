import json
import Twython
import tweepy



credentials = None
with open("twitter_credentials.json", "r") as file:
    credentials = json.load(file)

auth = tweepy.OAuthHandler(credentials['CONSUMER_KEY'], credentials['CONSUMER_SECRET'])
client = tweepy.Client() # Twitter API v2



def search_tweets(search, max=5000):
    tweets = client.search_recent_tweets(search)
    return [status['text'] for status in tweets['statuses']]


def create_tweets_file_for_hashtag(hashtag, max=5000, path=None):
    tweets = search_tweets('#' + hashtag, max=max)
    if path is None:
        path = 'data/hastags/' + hashtag
    with open(path, 'w') as file:
        file.writelines(tweets)
    


# api = Twython(credentials['CONSUMER_KEY'], credentials['CONSUMER_SECRET'])

# def search_tweets(search, max=5000):
#     tweets = api.search({
#         'q': search,
#         'result_type': 'popular',
#         'count': max,
#         'lang': 'pt',
#     })
#     return [status['text'] for status in tweets['statuses']]