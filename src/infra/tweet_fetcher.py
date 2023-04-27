import json
import tweepy
from preprocessing.text_transforms import TextTransforms
from infra.utils import create_dataset_folder


credentials = None
with open("twitter_credentials.json", "r") as file:
    credentials = json.load(file)

BEARER_TOKEN = credentials['BEARER_TOKEN']



class TweetFetcher(object):


    def __init__(self, hashtag, page_size=50):
        self._client = tweepy.Client(BEARER_TOKEN)
        self._hashtag = hashtag
        self._page_size = page_size
        self._next_token = None
        self._first_page_fetched = False



    def next_page(self) -> None:
        response = self._client.search_recent_tweets(
            query=f'#{self._hashtag}', 
            max_results=self._page_size, 
            next_token=self._next_token
        )
        
        texts = [tweet for tweet in response.data]
        
        if 'next_token' in response.meta:
            self._next_token = response.meta['next_token']
        
        self._first_page_fetched = True
        
        return texts



    def can_fetch_more(self) -> bool:
        return not self._first_page_fetched or not self._next_token is None



# def create_tweets_file_for_hashtag(hashtag: str, max: int = 50, path: str = None) -> None:
#     """
#     Creates a file containing tweets that contain a specified hashtag.
    
#     Parameters
#     ----------
#         hashtag : str 
#             A string that represents the hashtag to search for.
#         max : int 
#             An integer that represents the maximum number of tweets to retrieve (default is 50).
#         path : str 
#             A string that represents the file path where the tweets will be saved.

#     Returns
#     ----------
#         None : This function does not return anything. Instead, it saves the tweets in a file specified by the path parameter.
#     """
    
#     transforms = TextTransforms()
    
#     create_dataset_folder(hashtag)
    
#     raw_tweets_file_path = f'{path}/{hashtag}'
#     lemmatized_tweets_file_path = f'{path}/{hashtag}.lem'

#     with open(raw_tweets_file_path, 'a+') as raw_tweets_file, open(lemmatized_tweets_file_path, 'a+') as lemmatized_tweets_file:
        
#         tweets = tweepy.Cursor(api.search, q=hashtag).items(max)
        
#         for tweet in tweets:
#             raw = tweet.text.replace('\n', ' ') + '\n'
#             lemmatized = transforms.lemmatize(raw)
            
#             if lemmatized:
#                 raw_tweets_file.write(raw)
#                 lemmatized_tweets_file.write(lemmatized)

