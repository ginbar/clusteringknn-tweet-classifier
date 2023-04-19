import argparse
from infra.tweet_fetcher import TweetFetcher 
from preprocessing.text_transforms import TextTransforms
from io.dataset_writer import DatasetWriter

argument_parser = argparse.ArgumentParser("Create dataset")

argument_parser.add_argument("hashtag", help="Hashtag to be searched.", type=str, required=True)
argument_parser.add_argument("maximum", help="Maximum number of tweets to be fetched.", type=int, required=True)
argument_parser.add_argument("train_percent", help="Percentage of the data to be used for training.", type=float, required=False, default=0.8)

args = argument_parser.parse_args()

hashtag = args.hashtag
fetcher = TweetFetcher()
transforms = TextTransforms()

tweet_cursor = fetcher.get_cursor_for_hashtag(hashtag, max=args.maximum)

with DatasetWriter(hashtag, 'train') as train_dataset, DatasetWriter(hashtag, 'test') as test_dataset:
    
    max_train_dataset_size = args.maximum * args.train_percent
    iterated_tweets = 0

    for tweet in tweet_cursor:
        raw = transforms.remove_inner_newline_chars(tweet.text)
        lemmatized = transforms.lemmatize(raw)
        
        if lemmatized:
            if iterated_tweets <= max_train_dataset_size:
                train_dataset.save_raw_tweet(raw)
                train_dataset.save_lemmatized_tweet(lemmatized)
            else:
                test_dataset.save_raw_tweet(raw)
                test_dataset.save_lemmatized_tweet(lemmatized)
        