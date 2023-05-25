import argparse
import os
from infra.dataset_reader import DatasetReader
from infra.dataset_writer import DatasetWriter


argument_parser = argparse.ArgumentParser("Create dataset")

argument_parser.add_argument("hashtag", help="Hashtag to be searched.", type=str)

args = argument_parser.parse_args()

hashtag = args.hashtag

reader = DatasetReader(args.hashtag, 'train')

tweets = reader.get_lemmatized_tweets()

tweets = set(tweets)

print(len(tweets))

with open(os.open(f'../{hashtag}.cleaned', os.O_CREAT | os.O_WRONLY) , 'w') as cleaned_file:
    cleaned_file.writelines(tweets)
