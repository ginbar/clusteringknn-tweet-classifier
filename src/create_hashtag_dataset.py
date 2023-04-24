import argparse
from infra.tweet_fetcher import TweetFetcher 
from infra.dataset_writer import DatasetWriter
from preprocessing.text_transforms import TextTransforms

argument_parser = argparse.ArgumentParser("Create dataset")

argument_parser.add_argument("hashtag", help="Hashtag to be searched.", type=str)
argument_parser.add_argument("maximum", help="Maximum number of tweets to be fetched.", type=int)
argument_parser.add_argument("--percentage", help="Percentage of the data to be used for training.", type=float, default=0.8)
argument_parser.add_argument("--language", help="Data language.", type=str, default='portuguese')

args = argument_parser.parse_args()

hashtag = args.hashtag

fetcher = TweetFetcher(hashtag, page_size=10)
transforms = TextTransforms(language=args.language)


with DatasetWriter(hashtag, 'train') as train_dataset, DatasetWriter(hashtag, 'test') as test_dataset:
    
    max_train_dataset_size = args.maximum * args.percentage
    counter = 0

    while fetcher.can_fetch_more() and counter < args.maximum:
        
        tweets = fetcher.next_page()
        
        for tweet in tweets:

            raw = transforms.remove_inner_newline_chars(tweet.text)
            lemmatized = transforms.lemmatize(raw)
            
            if lemmatized:
                
                if counter <= max_train_dataset_size:
                    train_dataset.save_raw_tweet(raw)
                    train_dataset.save_lemmatized_tweet(lemmatized)
                else:
                    test_dataset.save_raw_tweet(raw)
                    test_dataset.save_lemmatized_tweet(lemmatized)
                
                counter += 1
            